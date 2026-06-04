import streamlit as st
import torch
import torch.nn as nn
import faiss
import pandas as pd

from pathlib import Path
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0


st.set_page_config(
    page_title="Nearly",
    layout="wide"
)

st.title("Nearly")
st.write("Find visually similar products within your budget.")


MODEL_PATH = "models/nearly_fashion_model.pth"
INDEX_PATH = "artifacts/fashion_index.faiss"
NAMES_PATH = "artifacts/fashion_image_names.txt"
METADATA_PATH = "data/subset/train_styles.csv"
IMAGE_DIR = Path("data/subset/images")


@st.cache_resource
def load_model():
    model = efficientnet_b0(weights=None)

    model.classifier[1] = nn.Linear(
        in_features=1280,
        out_features=13
    )

    model.load_state_dict(
        torch.load(MODEL_PATH, map_location="cpu")
    )

    model.classifier = nn.Identity()
    model.eval()

    return model


@st.cache_resource
def load_index():
    return faiss.read_index(INDEX_PATH)


@st.cache_data
def load_image_names():
    with open(NAMES_PATH, "r") as f:
        return [line.strip() for line in f.readlines()]


@st.cache_data
def load_metadata():
    metadata = pd.read_csv(METADATA_PATH)
    metadata["filename"] = metadata["id"].astype(str) + ".jpg"
    return metadata


model = load_model()
index = load_index()
image_names = load_image_names()
metadata = load_metadata()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


uploaded_file = st.file_uploader(
    "Upload a fashion image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    query_image = Image.open(uploaded_file).convert("RGB")

    st.image(
        query_image,
        caption="Uploaded Image",
        width=300
    )

    image_tensor = transform(query_image).unsqueeze(0)

    with torch.no_grad():
        embedding = model(image_tensor)

    query_embedding = embedding.cpu().numpy().astype("float32")
    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(query_embedding, 6)

    st.success("Search complete!")
    st.subheader("🎯 Top Similar Products")

    cols = st.columns(5)

    for col, idx, score in zip(
        cols,
        indices[0][1:],
        scores[0][1:]
    ):
        image_name = image_names[idx]
        image_path = IMAGE_DIR / image_name

        row = metadata[metadata["filename"] == image_name].iloc[0]

        similarity = int(score * 100)

        with col:
            st.image(
                str(image_path),
                use_container_width=True
            )

            st.markdown(f"**Similarity:** {similarity}%")
            st.markdown(f"**{row['productDisplayName']}**")
            st.write(f"Type: {row['articleType']}")
            st.write(f"Category: {row['subCategory']}")
            st.write(f"Color: {row['baseColour']}")