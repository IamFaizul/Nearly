from pathlib import Path

import faiss
import numpy as np


# ----------------------------
# Paths
# ----------------------------
ARTIFACTS_DIR = Path("artifacts")

EMBEDDINGS_PATH = ARTIFACTS_DIR / "sample_embeddings.npy"
INDEX_PATH = ARTIFACTS_DIR / "sample_index.faiss"


# ----------------------------
# Load embeddings
# ----------------------------
embeddings = np.load(EMBEDDINGS_PATH).astype("float32")

print(f"Loaded embeddings shape: {embeddings.shape}")


# ----------------------------
# Normalize embeddings
# ----------------------------
faiss.normalize_L2(embeddings)


# ----------------------------
# Build FAISS index
# ----------------------------
dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(embeddings)


# ----------------------------
# Save index
# ----------------------------
faiss.write_index(index, str(INDEX_PATH))

print(f"FAISS index built.")
print(f"Index dimension: {dimension}")
print(f"Total vectors in index: {index.ntotal}")
print(f"Saved to: {INDEX_PATH}")