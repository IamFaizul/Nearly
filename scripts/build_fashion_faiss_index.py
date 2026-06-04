from pathlib import Path

import faiss
import numpy as np


ARTIFACTS_DIR = Path("artifacts")

EMBEDDINGS_PATH = ARTIFACTS_DIR / "fashion_embeddings.npy"
INDEX_PATH = ARTIFACTS_DIR / "fashion_index.faiss"


embeddings = np.load(EMBEDDINGS_PATH).astype("float32")

print("Loaded embeddings:", embeddings.shape)

faiss.normalize_L2(embeddings)

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(embeddings)

faiss.write_index(index, str(INDEX_PATH))

print("Fashion FAISS index built.")
print("Index dimension:", dimension)
print("Total vectors:", index.ntotal)
print("Saved:", INDEX_PATH)