from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

# Path to subset images
IMAGES_DIR = Path("data/subset/images")

# Get first image
image_path = next(IMAGES_DIR.glob("*.jpg"))

print(f"Loading image: {image_path.name}")

# Open image
image = Image.open(image_path)

# Print image info
print(f"Image size: {image.size}")
print(f"Image mode: {image.mode}")

# Display image
plt.imshow(image)
plt.title(image_path.name)
plt.axis("off")

plt.show()