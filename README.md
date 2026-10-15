# My GenAI Image Model

A complete starter project for building a custom text-to-image model.

There are two tracks:

1. `src/scratch/` — a small educational diffusion model implemented in PyTorch.
2. `src/fine_tune/` — practical LoRA fine-tuning of a pretrained diffusion model on your own image/caption dataset.

The scratch model is for learning the internals; the fine-tuning track is the practical route to a useful custom generator.

## Setup

Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt
```

For NVIDIA GPU users, install the CUDA-compatible PyTorch build from the official PyTorch instructions, then install the remaining requirements.

## Dataset

Put training images in:

```text
data/images/
```

Create `data/captions.jsonl`:

```json
{"file_name":"cat1.jpg","text":"a white cat sitting on a wooden chair"}
{"file_name":"city1.jpg","text":"a futuristic city at night"}
```

## Practical custom model

Train a LoRA adapter:

```bash
python -m src.fine_tune.train_lora \
  --pretrained_model "runwayml/stable-diffusion-v1-5" \
  --image_dir data/images \
  --captions data/captions.jsonl \
  --output_dir outputs/my_lora \
  --resolution 512 \
  --train_batch_size 1 \
  --max_train_steps 1000
```

Generate HD Images:

```bash
python -m src.fine_tune.generate \
  --base_model "runwayml/stable-diffusion-v1-5" \
  --lora outputs/my_lora \
  --prompt "a futuristic Indian city at night, cinematic, highly detailed" \
  --height 512 \
  --width 768 \
  --upscale_factor 2 \
  --output outputs/generated_hd.png
```

Launch the Web UI:

```bash
python app.py --base_model "runwayml/stable-diffusion-v1-5" --lora outputs/my_lora
```

## Educational scratch model

The scratch implementation uses a tiny text-conditioned diffusion U-Net. It is intentionally small for learning the mechanics of diffusion models.

Train:

```bash
python -m src.scratch.train --image_dir data/toy/images --captions data/toy/captions.jsonl --epochs 10 --size 32
```

Generate / Sample in HD:

```bash
python -m src.scratch.sample \
  --checkpoint outputs/scratch/model.pt \
  --prompt "red car" \
  --target_resolution 512 \
  --output outputs/scratch/generated_hd.png
```

## Architecture

```text
prompt
  ↓
tokenizer
  ↓
text embedding
  ↓
cross-attention
  ↓
diffusion U-Net
  ↑
noisy image + timestep
  ↓
predicted noise
  ↓
DDPM reverse process
  ↓
image
```

For a serious system, move from the educational pixel model toward a latent-diffusion/DiT architecture, then add distributed training, data quality pipelines, evaluation, safety, and inference optimization.
