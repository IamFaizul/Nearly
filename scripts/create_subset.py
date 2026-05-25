import pandas as pd
from pathlib import Path
import shutil

# Paths
RAW_DIR = Path("data/raw")
IMAGES_DIR = RAW_DIR / "images"
SUBSET_DIR = Path("data/subset/images")

# Create output folder
SUBSET_DIR.mkdir(parents=True, exist_ok=True)

# Load metadata
df = pd.read_csv(RAW_DIR / "styles.csv", on_bad_lines="skip")

# Keep only important categories
target_categories = ["Apparel", "Accessories", "Footwear"]

df = df[df["masterCategory"].isin(target_categories)]

# Keep only rows where image exists
df["image_path"] = df["id"].astype(str) + ".jpg"

df = df[df["image_path"].apply(lambda x: (IMAGES_DIR / x).exists())]

# Sample 3000 rows
df_subset = df.sample(n=3000, random_state=42)

print(f"Selected rows: {len(df_subset)}")

# Copy images
for _, row in df_subset.iterrows():
    src = IMAGES_DIR / row["image_path"]
    dst = SUBSET_DIR / row["image_path"]

    shutil.copy2(src, dst)

# Save subset metadata
df_subset.to_csv("data/subset/subset_styles.csv", index=False)

print("Subset creation complete.")