from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class FashionDataset(Dataset):
    def __init__(self, csv_path, image_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.image_dir = Path(image_dir)
        self.transform = transform

        self.classes = sorted(self.df["subCategory"].unique())

        self.label_to_idx = {
            label: idx for idx, label in enumerate(self.classes)
        }

        self.idx_to_label = {
            idx: label for label, idx in self.label_to_idx.items()
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_name = str(row["id"]) + ".jpg"
        image_path = self.image_dir / image_name

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label_name = row["subCategory"]
        label = self.label_to_idx[label_name]

        return image, label