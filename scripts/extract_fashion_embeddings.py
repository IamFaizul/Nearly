from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0


IMAGE_DIR = Path("data/subset/images")
CSV_PATH = Path("data/subset/train_styles.csv")
MODEL_PATH = Path("models/nearly_fashion_model.pth")
ARTIFACTS_DIR = Path("artifacts")

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


# Load metadata
df = pd.read_csv(CSV_PATH)

image_names = (df["id"].astype(str) + ".jpg").tolist()


# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# Load trained model
model = efficientnet_b0(weights=None)

model.classifier[1] = nn.Linear(
    in_features=1280,
    out_features=13
)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location="cpu")
)

# Remove classifier to output embeddings
model.classifier = nn.Identity()
model.eval()


embeddings = []
valid_image_names = []

with torch.no_grad():
    for i, image_name in enumerate(image_names, start=1):
        image_path = IMAGE_DIR / image_name

        if not image_path.exists():
            print(f"Missing image: {image_name}")
            continue

        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0)

        embedding = model(image_tensor)
        embedding = embedding.squeeze(0).numpy()

        embeddings.append(embedding)
        valid_image_names.append(image_name)

        if i % 100 == 0:
            print(f"Processed {i}/{len(image_names)} images")


embeddings = np.array(embeddings).astype("float32")

np.save(
    ARTIFACTS_DIR / "fashion_embeddings.npy",
    embeddings
)

with open(
    ARTIFACTS_DIR / "fashion_image_names.txt",
    "w"
) as f:
    for name in valid_image_names:
        f.write(name + "\n")


print("\nDone.")
print("Embeddings shape:", embeddings.shape)
print("Saved: artifacts/fashion_embeddings.npy")
print("Saved: artifacts/fashion_image_names.txt")