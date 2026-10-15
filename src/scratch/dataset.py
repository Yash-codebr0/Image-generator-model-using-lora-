import json
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class CaptionImageDataset(Dataset):
    def __init__(self, image_dir, captions_file, tokenizer, size=32):
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
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        image = Image.open(self.image_dir / row["file_name"]).convert("RGB")
        return self.tf(image), torch.tensor(self.tokenizer.encode(row["text"]), dtype=torch.long)
