from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

# Image path
IMAGES_DIR = Path("data/subset/images")

image_path = next(IMAGES_DIR.glob("*.jpg"))

print(f"Loading image: {image_path.name}")

# Open image
image = Image.open(image_path).convert("RGB")

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Apply transforms
image_tensor = transform(image)

# Print tensor info
print(f"Tensor shape: {image_tensor.shape}")
print(f"Tensor dtype: {image_tensor.dtype}")
print(f"Min value: {image_tensor.min()}")
print(f"Max value: {image_tensor.max()}")

# Display original image
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")

plt.show()