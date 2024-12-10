import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import streamlit as st
import torch
import clip
import numpy as np
from PIL import Image
import faiss
import os

# Load CLIP model and preprocess function
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Load pre-generated embeddings and image paths
embeddings = np.load("embeddings.npy")
image_paths = np.load("image_paths.npy")

# Create a FAISS index for efficient similarity search
index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 distance for similarity search
index.add(embeddings)

# Streamlit app layout
st.title("E-commerce Image Search")
st.write("Upload an image to find similar products.")

# Image upload section
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display the uploaded image
    st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

    # Preprocess the uploaded image and convert it to an embedding
    image = Image.open(uploaded_image)
    image_input = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        query_embedding = model.encode_image(image_input).cpu().numpy()

    # Perform similarity search using FAISS
    D, I = index.search(query_embedding, k=5)  # Get the top 5 most similar images

    st.write(f"Top 5 similar products:")

    for i in range(5):
        st.image(image_paths[I[0][i]], caption=f"Similarity: {D[0][i]:.2f}", use_container_width=True)
        st.write(f"Product Image Path: {image_paths[I[0][i]]}")
