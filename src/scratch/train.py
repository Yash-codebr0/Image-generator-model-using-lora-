import argparse, json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .model import TinyTextConditionedUNet
from .tokenizer import SimpleTokenizer
from .dataset import CaptionImageDataset
from .diffusion import GaussianDiffusion

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image_dir", default="data/toy/images")
    p.add_argument("--captions", default="data/toy/captions.jsonl")
    p.add_argument("--output", default="outputs/scratch/model.pt")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--size", type=int, default=32)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    with open(args.captions, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    tok = SimpleTokenizer()
    tok.build([r["text"] for r in rows])

    ds = CaptionImageDataset(args.image_dir, args.captions, tok, args.size)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = TinyTextConditionedUNet(vocab_size=tok.vocab_size).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    diffusion = GaussianDiffusion(device=device)

    model.train()
    for epoch in range(args.epochs):
        total = 0.0
        for images, ids in tqdm(dl, desc=f"epoch {epoch+1}"):
            images, ids = images.to(device), ids.to(device)
            t = torch.randint(0, diffusion.steps, (images.size(0),), device=device)
            noisy, noise = diffusion.q_sample(images, t)
            pred = model(noisy, t, ids)
            loss = ((pred - noise) ** 2).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()

        print(f"epoch={epoch+1} loss={total/max(len(dl),1):.5f}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "tokenizer": tok.state_dict(),
        "size": args.size,
        "config": {
            "vocab_size": tok.vocab_size,
            "size": args.size,
        }
    }, out)
    print(f"saved: {out}")

if __name__ == "__main__":
    main()
