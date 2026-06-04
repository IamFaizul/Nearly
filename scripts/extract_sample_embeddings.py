from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


# ----------------------------
# Config
# ----------------------------
IMAGES_DIR = Path("data/subset/images")
OUTPUT_DIR = Path("artifacts")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_IMAGES = 100


# ----------------------------
# Load model
# ----------------------------
weights = EfficientNet_B0_Weights.DEFAULT
model = efficientnet_b0(weights=weights)
model.classifier = torch.nn.Identity()
model.eval()


# ----------------------------
# Define transforms
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ----------------------------
# Collect image paths
# ----------------------------
image_paths = list(IMAGES_DIR.glob("*.jpg"))[:MAX_IMAGES]

embeddings = []
image_names = []


# ----------------------------
# Extract embeddings
# ----------------------------
with torch.no_grad():
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")

        image_tensor = transform(image)
        image_tensor = image_tensor.unsqueeze(0)

        embedding = model(image_tensor)

        embedding = embedding.squeeze(0).numpy()

        embeddings.append(embedding)
        image_names.append(image_path.name)

        print(f"Processed: {image_path.name}")


# ----------------------------
# Save outputs
# ----------------------------
embeddings = np.array(embeddings).astype("float32")

np.save(OUTPUT_DIR / "sample_embeddings.npy", embeddings)

with open(OUTPUT_DIR / "sample_image_names.txt", "w") as f:
    for name in image_names:
        f.write(name + "\n")


print("\nDone.")
print(f"Embeddings shape: {embeddings.shape}")
print(f"Saved to: {OUTPUT_DIR / 'sample_embeddings.npy'}")

