import os
import io
import base64
import urllib.parse
from typing import Optional
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
from PIL import Image

app = FastAPI(
    title="GenAI Image Generator",
    description="FastAPI text-to-image generation app, deployable on Vercel and local servers.",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Check if local PyTorch / Diffusers pipeline is available
LOCAL_PIPELINE = None
try:
    import torch
    from diffusers import StableDiffusionPipeline
    
    BASE_MODEL = os.getenv("BASE_MODEL", "")
    LORA_PATH = os.getenv("LORA_PATH", "")
    
    if BASE_MODEL:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        pipe = StableDiffusionPipeline.from_pretrained(BASE_MODEL, torch_dtype=dtype)
        if LORA_PATH and os.path.exists(LORA_PATH):
            pipe.load_lora_weights(LORA_PATH)
        LOCAL_PIPELINE = pipe.to(device)
except Exception:
    LOCAL_PIPELINE = None


class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = ""
    width: Optional[int] = 512
    height: Optional[int] = 512
    steps: Optional[int] = 30
    guidance_scale: Optional[float] = 7.5
    seed: Optional[int] = 42
    model: Optional[str] = "flux"


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "engine": "local_pytorch" if LOCAL_PIPELINE is not None else "cloud_inference",
        "platform": "vercel" if os.getenv("VERCEL") else "standalone"
    }


def generate_cloud_image(prompt: str, negative_prompt: str, width: int, height: int, seed: int, model: str = "flux") -> str:
    """
    High-speed, serverless-friendly cloud image generator (Pollinations.ai / Hugging Face).
    Zero-config, free, and operates within Vercel's serverless timeout and size limits.
    """
    clean_prompt = prompt.strip()
    if negative_prompt:
        clean_prompt += f" (avoid: {negative_prompt.strip()})"
    
    encoded_prompt = urllib.parse.quote(clean_prompt)
    image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width={width}&height={height}&seed={seed}&model={model}&nologo=true"
    
    # Verify image generation
    resp = requests.get(image_url, timeout=25)
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail="Cloud inference failed to return an image.")
    
    # Return as base64 data URI so the client receives the self-contained image immediately
    b64_image = base64.b64encode(resp.content).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_image}"


@app.post("/api/generate")
@app.post("/generate")
def generate_image(payload: GenerateRequest):
    if not payload.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")
    
    try:
        if LOCAL_PIPELINE is not None:
            # Local PyTorch inference if available
            import torch
            generator = torch.Generator(device=LOCAL_PIPELINE.device).manual_seed(payload.seed or 42)
            result = LOCAL_PIPELINE(
                prompt=payload.prompt,
                negative_prompt=payload.negative_prompt,
                width=payload.width,
                height=payload.height,
                num_inference_steps=payload.steps,
                guidance_scale=payload.guidance_scale,
                generator=generator
            ).images[0]
            
            buf = io.BytesIO()
            result.save(buf, format="PNG")
            b64_img = base64.b64encode(buf.getvalue()).decode("utf-8")
            data_uri = f"data:image/png;base64,{b64_img}"
        else:
            # Cloud inference for Vercel / Serverless deployments
            data_uri = generate_cloud_image(
                prompt=payload.prompt,
                negative_prompt=payload.negative_prompt or "",
                width=payload.width or 512,
                height=payload.height or 512,
                seed=payload.seed or 42,
                model=payload.model or "flux"
            )
            
        return {
            "status": "success",
            "prompt": payload.prompt,
            "seed": payload.seed,
            "image": data_uri
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", response_class=HTMLResponse)
def index_page():
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GenAI Image Generator</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #0a0e17;
            --bg-card: #121826;
            --bg-input: #1a2234;
            --border: #26334d;
            --accent: #6366f1;
            --accent-hover: #4f46e5;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', sans-serif; }
        body {
            background-color: var(--bg-primary);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 2rem 1rem;
        }
        .container {
            max-width: 1050px;
            width: 100%;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
            margin-top: 1.5rem;
        }
        @media (max-width: 860px) {
            .container { grid-template-columns: 1fr; }
        }
        header { text-align: center; margin-bottom: 1rem; }
        header h1 { font-size: 2.2rem; font-weight: 700; background: linear-gradient(135deg, #a5b4fc, #6366f1); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        header p { color: var(--text-muted); margin-top: 0.5rem; font-size: 0.95rem; }
        .card {
            background-color: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 1.75rem;
            box-shadow: 0 10px 25px rgba(0,0,0,0.4);
        }
        .form-group { margin-bottom: 1.25rem; }
        label { display: block; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-muted); margin-bottom: 0.5rem; }
        textarea, select, input[type="number"] {
            width: 100%;
            background: var(--bg-input);
            border: 1px solid var(--border);
            color: var(--text-main);
            padding: 0.75rem 1rem;
            border-radius: 8px;
            font-size: 0.95rem;
            outline: none;
            transition: border-color 0.2s;
        }
        textarea:focus, select:focus, input[type="number"]:focus {
            border-color: var(--accent);
        }
        textarea { resize: vertical; min-height: 90px; }
        .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
        button.btn-generate {
            width: 100%;
            padding: 0.9rem 1.5rem;
            background: var(--accent);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
        }
        button.btn-generate:hover:not(:disabled) { background: var(--accent-hover); transform: translateY(-1px); }
        button.btn-generate:disabled { opacity: 0.6; cursor: not-allowed; }
        .preview-box {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 420px;
            border: 2px dashed var(--border);
            border-radius: 12px;
            overflow: hidden;
            position: relative;
            background: rgba(18, 24, 38, 0.6);
        }
        .preview-box img {
            width: 100%;
            height: auto;
            max-height: 520px;
            object-fit: contain;
            border-radius: 8px;
        }
        .spinner {
            display: none;
            width: 44px;
            height: 44px;
            border: 4px solid rgba(99, 102, 241, 0.2);
            border-top: 4px solid var(--accent);
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
        }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .empty-state { text-align: center; color: var(--text-muted); font-size: 0.9rem; padding: 2rem; }
        .action-bar { display: none; width: 100%; margin-top: 1rem; }
        .btn-download {
            width: 100%;
            background: #1e293b;
            border: 1px solid var(--border);
            color: var(--text-main);
            padding: 0.65rem;
            border-radius: 8px;
            font-weight: 500;
            cursor: pointer;
            text-align: center;
            text-decoration: none;
            display: block;
        }
        .btn-download:hover { background: #334155; }
        footer { margin-top: 2rem; font-size: 0.85rem; color: var(--text-muted); }
        footer a { color: var(--accent); text-decoration: none; }
    </style>
</head>
<body>
    <header>
        <h1>GenAI Image Studio</h1>
        <p>FastAPI Serverless Image Generation &mdash; Ready for Vercel</p>
    </header>

    <div class="container">
        <!-- Controls -->
        <div class="card">
            <div class="form-group">
                <label for="prompt">Prompt</label>
                <textarea id="prompt" placeholder="A futuristic cyberpunk city at night with neon reflections, 8k cinematic..."></textarea>
            </div>

            <div class="form-group">
                <label for="negative_prompt">Negative Prompt</label>
                <textarea id="negative_prompt" style="min-height: 60px;" placeholder="low quality, blurry, distorted, watermark..."></textarea>
            </div>

            <div class="grid-2">
                <div class="form-group">
                    <label for="resolution">Aspect Ratio</label>
                    <select id="resolution">
                        <option value="512x512" selected>Square (512 &times; 512)</option>
                        <option value="512x768">Portrait (512 &times; 768)</option>
                        <option value="768x512">Landscape (768 &times; 512)</option>
                        <option value="1024x1024">Square HD (1024 &times; 1024)</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="model">Model Engine</label>
                    <select id="model">
                        <option value="flux" selected>Flux (State of the art)</option>
                        <option value="turbo">SDXL Turbo (Ultra fast)</option>
                    </select>
                </div>
            </div>

            <div class="form-group">
                <label for="seed">Seed (-1 for random)</label>
                <input type="number" id="seed" value="-1">
            </div>

            <button id="generateBtn" class="btn-generate" onclick="handleGenerate()">
                <span>Generate Image</span>
            </button>
        </div>

        <!-- Output Preview -->
        <div class="card">
            <div class="preview-box" id="previewBox">
                <div class="spinner" id="spinner"></div>
                <div class="empty-state" id="emptyState">
                    <svg width="48" height="48" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24" style="margin-bottom: 0.75rem; opacity: 0.5;">
                        <path stroke-linecap="round" stroke-linejoin="round" d="m2.25 15.75 5.159-5.159a2.25 2.25 0 0 1 3.182 0l5.159 5.159m-1.5-1.5 1.409-1.409a2.25 2.25 0 0 1 3.182 0l2.909 2.909m-18 3.75h16.5a1.5 1.5 0 0 0 1.5-1.5V6a1.5 1.5 0 0 0-1.5-1.5H3.75A1.5 1.5 0 0 0 2.25 6v12a1.5 1.5 0 0 0 1.5 1.5Zm10.5-11.25h.008v.008h-.008V8.25Zm.375 0a.375.375 0 1 1-.75 0 .375.375 0 0 1 .75 0Z" />
                    </svg>
                    <p>Enter a prompt and click generate to view the result here.</p>
                </div>
                <img id="outputImg" style="display:none;" alt="Generated Image">
            </div>

            <div class="action-bar" id="actionBar">
                <a id="downloadLink" class="btn-download" download="generated_image.png">Download Image</a>
            </div>
        </div>
    </div>

    <footer>
        FastAPI Backend &bull; <a href="/docs" target="_blank">Interactive Swagger API Docs</a>
    </footer>

    <script>
        async function handleGenerate() {
            const prompt = document.getElementById('prompt').value.trim();
            if (!prompt) {
                alert('Please enter a prompt!');
                return;
            }

            const negative_prompt = document.getElementById('negative_prompt').value.trim();
            const resParts = document.getElementById('resolution').value.split('x');
            const width = parseInt(resParts[0]);
            const height = parseInt(resParts[1]);
            const model = document.getElementById('model').value;
            let seed = parseInt(document.getElementById('seed').value);
            if (isNaN(seed) || seed < 0) {
                seed = Math.floor(Math.random() * 1000000);
            }

            const btn = document.getElementById('generateBtn');
            const spinner = document.getElementById('spinner');
            const emptyState = document.getElementById('emptyState');
            const outputImg = document.getElementById('outputImg');
            const actionBar = document.getElementById('actionBar');
            const downloadLink = document.getElementById('downloadLink');

            btn.disabled = true;
            btn.innerText = "Generating...";
            emptyState.style.display = "none";
            outputImg.style.display = "none";
            actionBar.style.display = "none";
            spinner.style.display = "block";

            try {
                const res = await fetch('/api/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        prompt: prompt,
                        negative_prompt: negative_prompt,
                        width: width,
                        height: height,
                        seed: seed,
                        model: model
                    })
                });

                if (!res.ok) {
                    const errorData = await res.json().catch(() => ({ detail: 'Generation error' }));
                    throw new Error(errorData.detail || 'Failed to generate image');
                }

                const data = await res.json();
                outputImg.src = data.image;
                outputImg.style.display = "block";
                downloadLink.href = data.image;
                actionBar.style.display = "block";
            } catch (err) {
                alert('Error: ' + err.message);
                emptyState.style.display = "block";
            } finally {
                spinner.style.display = "none";
                btn.disabled = false;
                btn.innerText = "Generate Image";
            }
        }
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
