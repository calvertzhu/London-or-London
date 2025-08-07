import random
from pathlib import Path
from PIL import Image
import pandas as pd
import torchvision.transforms as T

from data_collection.config import DATA_DIR, METADATA_CSV_PATH

# Config
AUG_SUFFIX = "_aug"
AUG_PROB = 0.05  # 0.05% augmentation chance
AUG_SOURCE_LABEL = "augmented"

# Torchvision augmentation
torch_augment = T.Compose([
    T.RandomApply([T.GaussianBlur(kernel_size=5)], p=0.5),
    T.RandomApply([T.ColorJitter(brightness=0.3, contrast=0.3)], p=0.6),
    T.RandomApply([T.RandomRotation(degrees=(-15, 15))], p=0.2),
])

def augment_dataset(run_id=None):
    if not METADATA_CSV_PATH.exists():
        print(f"Metadata file not found: {METADATA_CSV_PATH}")
        return

    # Read full metadata once
    full_df = pd.read_csv(METADATA_CSV_PATH)
    
    # Get data to augment
    if run_id:
        if "run_id" not in full_df.columns:
            print("No 'run_id' column in metadata. Cannot filter.")
            return
        df = full_df[full_df["run_id"] == run_id]
        print(f"Filtering metadata by run_id={run_id}: {len(df)} rows to augment.")
    else:
        df = full_df

    new_rows = []
    skipped = 0

    for _, row in df.iterrows():
        if AUG_SUFFIX in row["filename"]:
            continue  # Already augmented

        if random.random() >= AUG_PROB:
            skipped += 1
            continue

        try:
            city, season, sharpness = row["city"], row["season"], row["sharpness"]
            src_path = DATA_DIR / city / season / sharpness / row["filename"]

            if not src_path.exists():
                print(f"Missing source image: {src_path}")
                continue

            image = Image.open(src_path).convert("RGB")
            image_aug = torch_augment(image)

            # Generate unique filename
            base = src_path.stem
            ext = src_path.suffix
            i = 1
            while True:
                aug_filename = f"{base}{AUG_SUFFIX}{i}{ext}"
                aug_path = src_path.parent / aug_filename
                if not aug_path.exists():
                    break
                i += 1

            image_aug.save(aug_path)
            print(f"Augmented: {row['filename']} → {aug_filename}")

            row_data = row.to_dict()
            row_data["filename"] = aug_filename
            row_data["source"] = AUG_SOURCE_LABEL
            if run_id:
                row_data["run_id"] = run_id
            new_rows.append(row_data)

        except Exception as e:
            print(f"Error augmenting {row['filename']}: {e}")

    if new_rows:
        # Create DataFrame with new augmented rows
        df_aug = pd.DataFrame(new_rows)
        
        if run_id:
            # If run_id specified, keep other runs unchanged
            df_combined = pd.concat([
                full_df[full_df["run_id"] != run_id],  # Keep other runs unchanged
                df,  # Keep original data from this run
                df_aug  # Add new augmented data
            ], ignore_index=True)
        else:
            # If no run_id, append all augmented data
            df_combined = pd.concat([full_df, df_aug], ignore_index=True)
            
        df_combined.to_csv(METADATA_CSV_PATH, index=False)
        print(f"\nAppended {len(new_rows)} augmented rows to {METADATA_CSV_PATH}")
    else:
        print("\nNo augmentations performed.")

    print(f"Skipped {skipped} images by chance.")

if __name__ == "__main__":
    augment_dataset()
