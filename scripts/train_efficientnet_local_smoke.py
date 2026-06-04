import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import transforms
from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights
)

from torch.utils.data import DataLoader
from torch.utils.data import random_split

from src.dataset import FashionDataset


# ----------------------------
# Config
# ----------------------------

BATCH_SIZE = 32
EPOCHS = 1
NUM_CLASSES = 13


# ----------------------------
# Dataset
# ----------------------------

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

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ----------------------------
# Model
# ----------------------------

weights = EfficientNet_B0_Weights.DEFAULT

model = efficientnet_b0(weights=weights)

model.classifier[1] = nn.Linear(
    in_features=1280,
    out_features=NUM_CLASSES
)

# ----------------------------
# Loss + Optimizer
# ----------------------------

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

# ----------------------------
# Training
# ----------------------------

model.train()

running_loss = 0

for images, labels in train_loader:

    optimizer.zero_grad()

    outputs = model(images)

    loss = criterion(
        outputs,
        labels
    )

    loss.backward()

    optimizer.step()

    running_loss += loss.item()

print(
    "Training Loss:",
    running_loss / len(train_loader)
)

# ----------------------------
# Validation
# ----------------------------

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in val_loader:

        outputs = model(images)

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

accuracy = 100 * correct / total

print(
    f"Validation Accuracy: {accuracy:.2f}%"
)