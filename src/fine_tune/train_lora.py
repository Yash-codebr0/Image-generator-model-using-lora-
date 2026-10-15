import argparse, json, math
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from accelerate import Accelerator
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model

class ImageCaptionDataset(Dataset):
    def __init__(self, image_dir, captions_file, tokenizer, resolution):
        self.image_dir = Path(image_dir)
        self.rows = []
        with open(captions_file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        self.rows.append(json.loads(line))
                    except Exception:
                        pass
        self.tokenizer = tokenizer
        self.tf = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        image = Image.open(self.image_dir / row["file_name"]).convert("RGB")
        ids = self.tokenizer(
            row["text"], padding="max_length", truncation=True,
            max_length=self.tokenizer.model_max_length, return_tensors="pt"
        ).input_ids[0]
        return {"pixel_values": self.tf(image), "input_ids": ids}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained_model", required=True)
    p.add_argument("--image_dir", required=True)
    p.add_argument("--captions", required=True)
    p.add_argument("--output_dir", default="outputs/my_lora")
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--train_batch_size", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--max_train_steps", type=int, default=1000)
    p.add_argument("--rank", type=int, default=8)
    args = p.parse_args()

    accelerator = Accelerator(mixed_precision="fp16" if torch.cuda.is_available() else "no")
    device = accelerator.device

    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model, subfolder="unet")
    scheduler = DDPMScheduler.from_pretrained(args.pretrained_model, subfolder="scheduler")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.05,
        bias="none",
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()

    ds = ImageCaptionDataset(args.image_dir, args.captions, tokenizer, args.resolution)
    dl = DataLoader(ds, batch_size=args.train_batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(
        [p for p in unet.parameters() if p.requires_grad],
        lr=args.learning_rate
    )

    unet, optimizer, dl = accelerator.prepare(unet, optimizer, dl)
    vae.to(device)
    text_encoder.to(device)
    vae.eval()
    text_encoder.eval()

    step = 0
    unet.train()
    progress = tqdm(total=args.max_train_steps, disable=not accelerator.is_local_main_process)

    while step < args.max_train_steps:
        for batch in dl:
            with accelerator.accumulate(unet):
                pixels = batch["pixel_values"].to(device)
                ids = batch["input_ids"].to(device)

                with torch.no_grad():
                    latents = vae.encode(pixels).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                    encoder_hidden_states = text_encoder(ids)[0]

                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, scheduler.config.num_train_timesteps,
                    (latents.shape[0],),
                    device=device
                ).long()
                noisy = scheduler.add_noise(latents, noise, timesteps)
                pred = unet(noisy, timesteps, encoder_hidden_states).sample
                loss = torch.nn.functional.mse_loss(pred.float(), noise.float())

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            step += 1
            progress.update(1)
            progress.set_postfix(loss=f"{loss.item():.4f}")
            if step >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        unwrapped = accelerator.unwrap_model(unet)
        unwrapped.save_pretrained(out)
        print(f"LoRA saved to {out}")

if __name__ == "__main__":
    main()
