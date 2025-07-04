import os
from PIL import Image

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "images"))
ROTATIONS = [90, 180, 270]

def augment_image(path):
    try:
        img = Image.open(path).convert("RGB")
        for angle in ROTATIONS:
            rotated = img.rotate(angle, expand=False)
            base, ext = os.path.splitext(path)
            new_path = f"{base}_rot{angle}{ext}"
            rotated.save(new_path, quality=90)
            print(f"Saved augmented image: {new_path}")
    except Exception as e:
        print(f"Failed to augment {path}: {e}")

def augment_dataset(data_dir):
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(".jpg"):
                full_path = os.path.join(root, file)
                augment_image(full_path)

if __name__ == "__main__":
    augment_dataset(DATA_DIR)
    print("✅ Done augmenting all images.")