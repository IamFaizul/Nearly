from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt


ARTIFACTS_DIR = Path("artifacts")
IMAGE_DIR = Path("data/subset/images")
CSV_PATH = Path("data/subset/train_styles.csv")

EMBEDDINGS_PATH = ARTIFACTS_DIR / "fashion_embeddings.npy"
INDEX_PATH = ARTIFACTS_DIR / "fashion_index.faiss"
NAMES_PATH = ARTIFACTS_DIR / "fashion_image_names.txt"


embeddings = np.load(EMBEDDINGS_PATH).astype("float32")
index = faiss.read_index(str(INDEX_PATH))

with open(NAMES_PATH, "r") as f:
    image_names = [line.strip() for line in f.readlines()]

metadata = pd.read_csv(CSV_PATH)
metadata["filename"] = metadata["id"].astype(str) + ".jpg"
metadata_lookup = metadata.set_index("filename").to_dict(orient="index")


query_index =1500  
query_embedding = embeddings[query_index:query_index + 1]

faiss.normalize_L2(query_embedding)

k = 6
scores, indices = index.search(query_embedding, k)


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