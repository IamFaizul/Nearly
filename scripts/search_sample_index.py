from pathlib import Path

import faiss
import numpy as np


# ----------------------------
# Paths
# ----------------------------
ARTIFACTS_DIR = Path("artifacts")

EMBEDDINGS_PATH = ARTIFACTS_DIR / "sample_embeddings.npy"
INDEX_PATH = ARTIFACTS_DIR / "sample_index.faiss"
NAMES_PATH = ARTIFACTS_DIR / "sample_image_names.txt"


# ----------------------------
# Load artifacts
# ----------------------------
embeddings = np.load(EMBEDDINGS_PATH).astype("float32")
index = faiss.read_index(str(INDEX_PATH))

with open(NAMES_PATH, "r") as f:
    image_names = [line.strip() for line in f.readlines()]


# ----------------------------
# Normalize query embedding
# ----------------------------
query_index = 0
query_embedding = embeddings[query_index:query_index + 1]

faiss.normalize_L2(query_embedding)


# ----------------------------
# Search
# ----------------------------
k = 5
scores, indices = index.search(query_embedding, k)


# ----------------------------
# Show results
# ----------------------------
print(f"Query image: {image_names[query_index]}")
print("\nTop matches:")

for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
    print(f"{rank}. {image_names[idx]} | similarity: {score:.4f}")