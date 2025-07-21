"""
dataset.py
-----------
Minimal loader that turns your CSV of
  file_name, caption
pairs into a PyTorch Dataset that yields
  { "pixel_values": tensor, "text": str }
suitable for the SDXL pipeline.

Assumes:
  project/
    ├── train.py
    ├── dataset.py   ← this file
    └── data/
        ├── metadata.csv
        └── *.jpg / *.png
"""

from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class TextImageDataset(Dataset):
    def __init__(self, csv_path, image_dir, resolution=512):
        self.df = pd.read_csv(csv_path)
        self.image_dir = Path(image_dir)
        self.transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),                 # → [0,1]
            transforms.Normalize([0.5], [0.5]),    # → [-1,1] expected by SDXL
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.image_dir / row["file_name"]
        if not img_path.is_file():
            raise FileNotFoundError(f"Image not found: {img_path}")

        img = Image.open(img_path).convert("RGB")
        return {
            "pixel_values": self.transform(img),
            "text": row["caption"]
        }
