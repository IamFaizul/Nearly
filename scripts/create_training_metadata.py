import pandas as pd

# Load metadata
df = pd.read_csv("data/subset/subset_styles.csv")

# Count samples per class
counts = df["subCategory"].value_counts()

# Keep classes with >= 50 samples
valid_classes = counts[counts >= 50].index

# Filter
df_train = df[df["subCategory"].isin(valid_classes)].copy()

print("Original rows:", len(df))
print("Filtered rows:", len(df_train))

print("\nClasses:")
print(df_train["subCategory"].value_counts())

# Save
df_train.to_csv(
    "data/subset/train_styles.csv",
    index=False
)

print("\nSaved:")
print("data/subset/train_styles.csv")