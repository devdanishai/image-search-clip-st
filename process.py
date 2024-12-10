# 1. for this file i need train folder with sub folders with picture
# and _tokenization.txt
# 2. this file will give 2 output files embeddings.npy
# and image_paths.npy

import os
import torch
import clip
from PIL import Image
import numpy as np

# Load CLIP Model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Directory of product images
image_folder = "train/"  # Update path

# Extract embeddings
embeddings = []
image_paths = []

# Recursively load images from subfolders
for root, _, files in os.walk(image_folder):
    for filename in files:
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(root, filename)
            image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
            with torch.no_grad():
                embedding = model.encode_image(image).cpu().numpy()
            embeddings.append(embedding)
            image_paths.append(image_path)

# Save embeddings and image paths
embeddings = np.vstack(embeddings)
np.save("embeddings.npy", embeddings)
np.save("image_paths.npy", np.array(image_paths))

print("Embeddings and image paths saved successfully!")
