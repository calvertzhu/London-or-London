import random
from pathlib import Path
from PIL import Image
import pandas as pd
import torchvision.transforms as T

from data_collection.config import DATA_DIR, METADATA_CSV_PATH

# Config
AUG_SUFFIX = "_aug"
AUG_PROB = 0.80  # 50% chance for testing — reduce to 0.05 in production
AUG_SOURCE_LABEL = "augmented"

# Define torchvision augmentation pipeline
torch_augment = T.Compose([
    T.RandomApply([T.GaussianBlur(kernel_size=5)], p=0.5),
    T.RandomApply([T.ColorJitter(brightness=0.3, contrast=0.3)], p=0.6),
    T.RandomApply([T.RandomRotation(degrees=(-15, 15))], p=0.2),
])

def augment_dataset(run_id=None):
    if not DATA_DIR.exists():
        print(f"Data directory not found: {DATA_DIR}")
        return
    if not METADATA_CSV_PATH.exists():
        print(f"Metadata file not found: {METADATA_CSV_PATH}")
        return

    df = pd.read_csv(METADATA_CSV_PATH)
    new_rows = []
    skipped = 0

    for image_path in DATA_DIR.rglob("*.jpg"):
        if AUG_SUFFIX in image_path.stem:
            continue  # Skip already augmented

        if random.random() >= AUG_PROB:
            skipped += 1
            continue  # Skip based on chance

        try:
            image = Image.open(image_path).convert("RGB")
            image_aug = torch_augment(image)

            # Generate unique augmented filename
            folder = image_path.parent
            base = image_path.stem
            ext = image_path.suffix
            i = 1
            while True:
                aug_filename = f"{base}{AUG_SUFFIX}{i}{ext}"
                aug_path = folder / aug_filename
                if not aug_path.exists():
                    break
                i += 1

            image_aug.save(aug_path)
            print(f"Augmented: {image_path.name} → {aug_path.name}")

            # Extract pano_id from filename
            parts = base.split("_")
            if len(parts) < 6:
                print(f"Unexpected filename format: {base}")
                continue

            pano_id = parts[0]
            matches = df[df["pano_id"].astype(str).str.strip() == pano_id.strip()]

            if matches.empty:
                print(f"No metadata found for {image_path.name} (pano_id={pano_id})")
                continue

            # Append augmented metadata row
            row_data = matches.iloc[0].to_dict()
            row_data["filename"] = aug_filename
            row_data["source"] = AUG_SOURCE_LABEL
            if run_id:
                row_data["run_id"] = run_id
            new_rows.append(row_data)

        except Exception as e:
            print(f"Error processing {image_path.name}: {e}")

    if new_rows:
        df_aug = pd.DataFrame(new_rows)
        df_combined = pd.concat([df, df_aug], ignore_index=True)
        df_combined.to_csv(METADATA_CSV_PATH, index=False)
        print(f"\nAppended {len(new_rows)} augmented rows to {METADATA_CSV_PATH}")
    else:
        print("\nNo augmentations performed.")

    print(f"Skipped {skipped} images by chance.")

if __name__ == "__main__":
    augment_dataset()
