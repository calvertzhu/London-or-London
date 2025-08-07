import os
import random
import shutil
from pathlib import Path

def print_all_files(path):
    """Print all files in directory tree for debugging"""
    path = Path(path)
    print(f"\nLooking in: {path}")
    for p in path.rglob("*"):
        if p.is_file():
            print(f"  - {p}")

def copy_images(images, split, city):
    """Copy images to their respective train/val directory"""
    for img in images:
        shutil.copy(img, target_dir / split / city / img.name)

# Setup directories
source_dir = Path("data")
target_dir = Path("report_data")
train_ratio = 0.7

# Print initial counts
print_all_files("data/london_on")
print_all_files("data/london_uk")

# Clean old directories if they exist
if (target_dir / "train").exists():
    shutil.rmtree(target_dir / "train")
if (target_dir / "val").exists():
    shutil.rmtree(target_dir / "val")

# Make new folders
for split in ["train", "val"]:
    for city in ["London_ON", "London_UK"]:
        (target_dir / split / city).mkdir(parents=True, exist_ok=True)

# Process each city
for city_src, city_dest in [("london_on", "London_ON"), ("london_uk", "London_UK")]:
    all_images = []
    # Collect all images from all seasons and sharpness levels
    for season_dir in (source_dir / city_src).iterdir():
        if season_dir.is_dir():
            for sharp_dir in season_dir.iterdir():
                if sharp_dir.is_dir():
                    all_images += list(sharp_dir.glob("*.*"))
    
    # Shuffle & split
    random.shuffle(all_images)
    train_count = int(train_ratio * len(all_images))
    train_imgs = all_images[:train_count]
    val_imgs = all_images[train_count:]
    
    # Copy to respective directories
    copy_images(train_imgs, "train", city_dest)
    copy_images(val_imgs, "val", city_dest)
    print(f"{city_dest}: {len(train_imgs)} train, {len(val_imgs)} val (of {len(all_images)})")

print("\nPhysical split complete. Check your 'report_data' folder.")
