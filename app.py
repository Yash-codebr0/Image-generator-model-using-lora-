import argparse
import torch
import gradio as gr
from diffusers import StableDiffusionPipeline

parser = argparse.ArgumentParser()
parser.add_argument("--base_model", required=True)
parser.add_argument("--lora", required=True)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

pipe = StableDiffusionPipeline.from_pretrained(args.base_model, torch_dtype=dtype)
pipe.load_lora_weights(args.lora)
pipe = pipe.to(device)

from PIL import Image

def generate(prompt, negative, steps, guidance, seed, preset, custom_w, custom_h, upscale_factor):
    preset_resolutions = {
        "Standard (512x512)": (512, 512),
        "Portrait (512x768)": (512, 768),
        "Landscape / HD Ready (768x512)": (768, 512),
        "Square HD (1024x1024)": (1024, 1024),
        "Widescreen HD 16:9 (1024x576)": (1024, 576),
    }

    if preset in preset_resolutions:
        w, h = preset_resolutions[preset]
    else:
        w, h = int(custom_w), int(custom_h)

    generator = torch.Generator(device=device).manual_seed(int(seed))
    image = pipe(
        prompt=prompt,
        negative_prompt=negative,
        height=h,
        width=w,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
        generator=generator
    ).images[0]

    upscale = int(upscale_factor)
    if upscale > 1:
        new_w = image.width * upscale
        new_h = image.height * upscale
        image = image.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)

    return image

demo = gr.Interface(
    fn=generate,
    inputs=[
        gr.Textbox(label="Prompt", placeholder="Enter your prompt here..."),
        gr.Textbox(label="Negative prompt", placeholder="low quality, blurry, distorted..."),
        gr.Slider(10, 60, value=30, step=1, label="Steps"),
        gr.Slider(1, 15, value=7.5, step=0.5, label="Guidance Scale (CFG)"),
        gr.Number(value=42, label="Seed"),
        gr.Dropdown(
            choices=[
                "Standard (512x512)",
                "Portrait (512x768)",
                "Landscape / HD Ready (768x512)",
                "Square HD (1024x1024)",
                "Widescreen HD 16:9 (1024x576)",
                "Custom",
            ],
            value="Standard (512x512)",
            label="Resolution Preset"
        ),
        gr.Slider(256, 1536, value=512, step=64, label="Custom Width (if Custom selected)"),
        gr.Slider(256, 1536, value=512, step=64, label="Custom Height (if Custom selected)"),
        gr.Radio(
            choices=[1, 2, 4],
            value=1,
            label="HD Upscale Factor (e.g. 2x = 2x resolution, 4x = Ultra HD)"
        ),
    ],
    outputs=gr.Image(type="pil", label="Generated Output"),
    title="My Custom GenAI Image Model (HD Supported)",
    description="Text-to-image generation using your trained LoRA adapter with High-Definition (HD) resolution and upscaling."
)

if __name__ == "__main__":
    demo.launch()
