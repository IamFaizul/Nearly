import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from torchvision import transforms
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

print("Dataset size:", len(dataset))
print("Classes:", dataset.classes)
print("Label mapping:", dataset.label_to_idx)

image, label = dataset[0]

print("Image tensor shape:", image.shape)
print("Label:", label)
print("Label name:", dataset.idx_to_label[label])