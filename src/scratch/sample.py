import argparse
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms
from .model import TinyTextConditionedUNet
from .tokenizer import SimpleTokenizer
from .diffusion import GaussianDiffusion

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="outputs/scratch/model.pt")
    p.add_argument("--prompt", required=True)
    p.add_argument("--output", default="outputs/scratch/generated.png")
    p.add_argument("--size", type=int, default=None, help="Base generation size (e.g. 32, 64)")
    p.add_argument("--height", type=int, default=None, help="Base generation height")
    p.add_argument("--width", type=int, default=None, help="Base generation width")
    p.add_argument("--target_resolution", type=int, default=None, help="HD target resolution (e.g. 512, 1024)")
    p.add_argument("--upscale_factor", type=int, default=1, help="Upscale factor multiplier (e.g. 2, 4, 8, 16)")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device)
    tok = SimpleTokenizer.from_state_dict(ckpt["tokenizer"])
    vocab_size = ckpt.get("config", {}).get("vocab_size", tok.vocab_size)
    model = TinyTextConditionedUNet(vocab_size=vocab_size).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    default_size = ckpt.get("size", 32)
    h = args.height or args.size or default_size
    w = args.width or args.size or default_size

    ids = torch.tensor([tok.encode(args.prompt)], device=device)
    diffusion = GaussianDiffusion(device=device)
    image = diffusion.sample(model, ids, (1, 3, h, w), device)[0]
    image = ((image + 1) * 127.5).byte().permute(1, 2, 0).cpu().numpy()

    pil_img = Image.fromarray(image)

    # High-Definition (HD) Resampling / Upscaling
    target_res = args.target_resolution
    if target_res is not None:
        pil_img = pil_img.resize((target_res, target_res), resample=Image.Resampling.LANCZOS)
    elif args.upscale_factor > 1:
        new_w = w * args.upscale_factor
        new_h = h * args.upscale_factor
        pil_img = pil_img.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pil_img.save(out)
    print(f"saved: {out} [Resolution: {pil_img.size[0]}x{pil_img.size[1]}]")

if __name__ == "__main__":
    main()
