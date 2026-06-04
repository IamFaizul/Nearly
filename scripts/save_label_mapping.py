import json
import pandas as pd
from pathlib import Path


ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv("data/subset/train_styles.csv")

classes = sorted(df["subCategory"].unique())

label_to_idx = {
    label: idx
    for idx, label in enumerate(classes)
}

with open("artifacts/label_mapping.json", "w") as f:
    json.dump(label_to_idx, f, indent=4)

print("Saved label mapping to artifacts/label_mapping.json")
print(label_to_idx)