import os
from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class SkinLesionDataset(Dataset):
    """
    Expects a CSV with columns: image_path, label
    image_path can be absolute or relative to root_dir.
    """

    def __init__(self, csv_file: str, root_dir: str = "", transform=None):
        self.df = pd.read_csv(csv_file)
        self.root_dir = Path(root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.root_dir / row["image_path"]
        image = Image.open(img_path).convert("RGB")
        label = int(row["label"])

        if self.transform:
            image = self.transform(image)

        return image, label
