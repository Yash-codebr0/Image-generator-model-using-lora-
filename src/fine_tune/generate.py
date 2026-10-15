import argparse
import torch
from diffusers import StableDiffusionPipeline

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", required=True)
    p.add_argument("--lora", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--negative_prompt", default="")
    p.add_argument("--output", default="outputs/generated.png")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--guidance", type=float, default=7.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--height", type=int, default=512, help="Generation height (e.g. 512, 768, 1024)")
    p.add_argument("--width", type=int, default=512, help="Generation width (e.g. 512, 768, 1024)")
    p.add_argument("--upscale_factor", type=int, default=1, help="Post-generation HD upscale factor (e.g. 2, 4)")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(args.base_model, torch_dtype=dtype)
    pipe.load_lora_weights(args.lora)
    pipe = pipe.to(device)

    generator = torch.Generator(device=device).manual_seed(args.seed)
    image = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        generator=generator
    ).images[0]

    if args.upscale_factor > 1:
        new_w = image.width * args.upscale_factor
        new_h = image.height * args.upscale_factor
        from PIL import Image
        image = image.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)

    image.save(args.output)
    print(f"saved: {args.output} [Resolution: {image.width}x{image.height}]")

if __name__ == "__main__":
    main()
