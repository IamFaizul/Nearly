import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from src.dataset import FashionDataset


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

dataset = FashionDataset(
    csv_path="data/subset/train_styles.csv",
    image_dir="data/subset/images",
    transform=transform
)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size]
)

print("Train size:", len(train_dataset))
print("Validation size:", len(val_dataset))

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)

images, labels = next(iter(train_loader))

print("\nTrain batch shape:", images.shape)
print("Train labels shape:", labels.shape)