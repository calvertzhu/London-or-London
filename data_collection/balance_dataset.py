import pandas as pd
from pathlib import Path
import shutil

# CONFIG
METADATA_PATH = Path("metadata/combined.csv")
BALANCED_CSV_PATH = Path("metadata/balanced_subset.csv")
BALANCED_IMAGE_DIR = Path("data_balanced/")
N_PER_CLASS = 1000
COPY_IMAGES = True  # set to False if you only want the CSV

# Load full metadata
df = pd.read_csv(METADATA_PATH)

# Sample n images per (city, season, sharpness)
balanced_df = (
    df.groupby(["city", "season", "sharpness"])
      .apply(lambda g: g.sample(n=min(N_PER_CLASS, len(g)), random_state=42))
      .reset_index(drop=True)
)

# Save balanced metadata
BALANCED_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
balanced_df.to_csv(BALANCED_CSV_PATH, index=False)
print(f"Saved balanced metadata to {BALANCED_CSV_PATH}")

# Copy image files to new folder
if COPY_IMAGES:
    for row in balanced_df.itertuples(index=False):
        src = Path("data") / row.city / row.season / row.sharpness / row.filename
        dst = BALANCED_IMAGE_DIR / row.city / row.season / row.sharpness / row.filename
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy(src, dst)
        except Exception as e:
            print(f"Failed to copy {src} → {dst}: {e}")
    print(f"📁 Copied balanced images to {BALANCED_IMAGE_DIR}")

# Summary
print("\nFinal sample counts:")
print(balanced_df.groupby(["city", "season", "sharpness"]).size())
