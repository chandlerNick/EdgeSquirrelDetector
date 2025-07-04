import os
from pyinaturalist import get_observations
from urllib.request import urlretrieve
from tqdm import tqdm

# Class label -> iNaturalist taxon ID
SPECIES = {
    "squirrel": [46017, 46018],  # Sciurus carolinensis, Sciurus griseus
    "no_squirrel": [12727, 13547, 559248, 7998]  # Robins, Chickadees, Sparrows, crows
}

# Output directory
OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "images"))
IMAGES_PER_SPECIES = 500


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def download_images(taxon_id, class_dir, count=100):
    page = 1
    downloaded = 0
    output_path = os.path.join(OUT_DIR, class_dir)
    ensure_dir(output_path)

    print(f"Attempting to download {count} images for taxon ID {taxon_id} → {class_dir}")

    with tqdm(total=count, desc=f"{class_dir} ({taxon_id})") as pbar:
        while downloaded < count:
            response = get_observations(
                taxon_id=taxon_id,
                quality_grade="research",
                has="photo",
                per_page=50,
                page=page
            )

            results = response.get("results", [])
            if not results:
                print(f"  ✖ No more results for taxon {taxon_id}")
                break

            for obs in results:
                if downloaded >= count:
                    break

                try:
                    photos = obs.get("photos", [])
                    if not photos:
                        continue
                    photo_url = photos[0]["url"].replace("square", "large")
                    filename = os.path.join(output_path, f"{taxon_id}_{obs['id']}.jpg")

                    if not os.path.exists(filename):
                        urlretrieve(photo_url, filename)
                        downloaded += 1
                        pbar.update(1)
                except Exception as e:
                    continue

            page += 1


def main():
    print("Downloading dataset...")
    for class_name, taxon_ids in SPECIES.items():
        for taxon_id in taxon_ids:
            download_images(taxon_id, class_name, count=IMAGES_PER_SPECIES)


if __name__ == "__main__":
    main()