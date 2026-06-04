from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt


# ----------------------------
# Paths
# ----------------------------
ARTIFACTS_DIR = Path("artifacts")
IMAGE_DIR = Path("data/subset/images")
METADATA_PATH = Path("data/subset/subset_styles.csv")


# ----------------------------
# Load artifacts
# ----------------------------
embeddings = np.load(ARTIFACTS_DIR / "sample_embeddings.npy").astype("float32")
index = faiss.read_index(str(ARTIFACTS_DIR / "sample_index.faiss"))

with open(ARTIFACTS_DIR / "sample_image_names.txt", "r") as f:
    image_names = [line.strip() for line in f.readlines()]


# ----------------------------
# Load metadata
# ----------------------------
metadata = pd.read_csv(METADATA_PATH)

# Create filename column for matching
metadata["filename"] = metadata["id"].astype(str) + ".jpg"

# Keep only metadata for sample images
metadata = metadata[metadata["filename"].isin(image_names)]

# Make quick lookup: filename -> metadata row
metadata_lookup = metadata.set_index("filename").to_dict(orient="index")


# ----------------------------
# Query
# ----------------------------
query_index = 0
query_embedding = embeddings[query_index:query_index + 1]

faiss.normalize_L2(query_embedding)

# Ask for 6 because first one will usually be the query itself
k = 6
scores, indices = index.search(query_embedding, k)


# ----------------------------
# Visualize
# ----------------------------
plt.figure(figsize=(14, 8))

for i, idx in enumerate(indices[0]):
    image_name = image_names[idx]
    image_path = IMAGE_DIR / image_name

    row = metadata_lookup.get(image_name, {})

    article_type = row.get("articleType", "Unknown")
    sub_category = row.get("subCategory", "Unknown")
    color = row.get("baseColour", "Unknown")

    image = Image.open(image_path).convert("RGB")

    plt.subplot(2, 3, i + 1)
    plt.imshow(image)

    if i == 0:
        title = (
            f"QUERY\n"
            f"{article_type}\n"
            f"{sub_category} | {color}"
        )
    else:
        title = (
            f"Match {i} | {scores[0][i]:.2f}\n"
            f"{article_type}\n"
            f"{sub_category} | {color}"
        )

    plt.title(title, fontsize=9)
    plt.axis("off")

plt.tight_layout()
plt.show()