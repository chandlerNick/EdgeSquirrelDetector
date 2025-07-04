# Resizes images to 224,224 to fit on the tiny edge models.

import os
from PIL import Image

TARGET_SIZE = (224, 224)
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "images"))

def resize_image(path):
    try:
        img = Image.open(path).convert("RGB")
        img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
        img.save(path, quality=90)
    except Exception as e:
        print(f"Failed to resize {path}: {e}")

def resize_dataset(data_dir):
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(".jpg"):
                full_path = os.path.join(root, file)
                resize_image(full_path)

if __name__ == "__main__":
    resize_dataset(DATA_DIR)
    print("✅ Done resizing all images.")
