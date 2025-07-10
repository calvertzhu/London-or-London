import pandas as pd
from pathlib import Path
import shutil

def balance_dataset(
    metadata_path="metadata/combined.csv",
    output_csv="metadata/balanced_subset.csv",
    output_dir="data_balanced/",
    n_per_class=1000,
    copy_images=True
):
    METADATA_PATH = Path(metadata_path)
    BALANCED_CSV_PATH = Path(output_csv)
    BALANCED_IMAGE_DIR = Path(output_dir)

    if not METADATA_PATH.exists():
        print(f"❌ Metadata file not found: {METADATA_PATH}")
        return

    df = pd.read_csv(METADATA_PATH)

    # Sample n images per (city, season, sharpness)
    balanced_df = (
        df.groupby(["city", "season", "sharpness"])
          .apply(lambda g: g.sample(n=min(n_per_class, len(g)), random_state=42))
          .reset_index(drop=True)
    )

    BALANCED_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    balanced_df.to_csv(BALANCED_CSV_PATH, index=False)
    print(f"✅ Saved balanced metadata to {BALANCED_CSV_PATH}")

    # Copy image files to new directory
    if copy_images:
        for row in balanced_df.itertuples(index=False):
            src = Path("data") / row.city / row.season / row.sharpness / row.filename
            dst = BALANCED_IMAGE_DIR / row.city / row.season / row.sharpness / row.filename
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy(src, dst)
            except Exception as e:
                print(f"⚠️ Failed to copy {src} → {dst}: {e}")
        print(f"📁 Copied balanced images to {BALANCED_IMAGE_DIR}")

    # Print summary
    print("\n📊 Final sample counts per class:")
    print(balanced_df.groupby(["city", "season", "sharpness"]).size())

if __name__ == "__main__":
    balance_dataset()
