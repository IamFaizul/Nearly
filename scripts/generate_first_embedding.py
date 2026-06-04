from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0
from torchvision.models import EfficientNet_B0_Weights


# 1. Load pretrained EfficientNet-B0
weights = EfficientNet_B0_Weights.DEFAULT
model = efficientnet_b0(weights=weights)

# 2. Replace classifier with Identity so output becomes embedding
model.classifier = torch.nn.Identity()

# 3. Set model to inference mode
model.eval()


# 4. Define preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 5. Load one image
image_path = next(Path("data/subset/images").glob("*.jpg"))
image = Image.open(image_path).convert("RGB")

print(f"Image: {image_path.name}")


# 6. Convert image to tensor
image_tensor = transform(image)

print(f"Image tensor shape: {image_tensor.shape}")


# 7. Add batch dimension
image_tensor = image_tensor.unsqueeze(0)

print(f"Batch tensor shape: {image_tensor.shape}")


# 8. Generate embedding
with torch.no_grad():
    embedding = model(image_tensor)

print(f"Embedding shape: {embedding.shape}")
print(f"First 10 embedding values:")
print(embedding[0][:10])


