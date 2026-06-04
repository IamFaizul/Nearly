from pathlib import Path

import faiss
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# ----------------------------
# Paths
# ----------------------------

ARTIFACTS_DIR = Path("artifacts")
IMAGE_DIR = Path("data/subset/images")

embeddings = np.load(
    ARTIFACTS_DIR / "sample_embeddings.npy"
).astype("float32")

index = faiss.read_index(
    str(ARTIFACTS_DIR / "sample_index.faiss")
)

with open(
    ARTIFACTS_DIR / "sample_image_names.txt"
) as f:
    image_names = [line.strip() for line in f]


# ----------------------------
# Query
# ----------------------------

query_index = 0

query_embedding = embeddings[
    query_index:query_index + 1
]

faiss.normalize_L2(query_embedding)

k = 6

scores, indices = index.search(
    query_embedding,
    k
)

# ----------------------------
# Visualize
# ----------------------------

plt.figure(figsize=(12, 8))

for i, idx in enumerate(indices[0]):

    image_name = image_names[idx]

    image_path = IMAGE_DIR / image_name

    image = Image.open(image_path)

    plt.subplot(2, 3, i + 1)

    plt.imshow(image)

    if i == 0:
        plt.title("QUERY")
    else:
        plt.title(
            f"Match {i}\n{scores[0][i]:.2f}"
        )

    plt.axis("off")

plt.tight_layout()

plt.show()