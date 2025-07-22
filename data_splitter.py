import os
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import random_split, ConcatDataset

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# DEBUG print out all files it sees
def print_all_files(path):
    path = Path(path)
    print(f"\nLooking in: {path}")
    for p in path.rglob("*"):
        if p.is_file():
            print(f"  - {p}")

print_all_files("data/london_on")
print_all_files("data/london_uk")

# Load datasets
london_on_dataset = datasets.ImageFolder(root="data/london_on", transform=transform)
london_uk_dataset = datasets.ImageFolder(root="data/london_uk", transform=transform)

print(f"\nLondon_ON total images found by ImageFolder: {len(london_on_dataset)}")
print(f"London_UK total images found by ImageFolder: {len(london_uk_dataset)}")

# Split London ON
on_train_len = int(0.7 * len(london_on_dataset))
on_val_len = len(london_on_dataset) - on_train_len
on_train, on_val = random_split(london_on_dataset, [on_train_len, on_val_len])

# Split London UK
uk_train_len = int(0.7 * len(london_uk_dataset))
uk_val_len = len(london_uk_dataset) - uk_train_len
uk_train, uk_val = random_split(london_uk_dataset, [uk_train_len, uk_val_len])

# Merge datasets
from torch.utils.data import DataLoader
train_dataset = ConcatDataset([on_train, uk_train])
val_dataset = ConcatDataset([on_val, uk_val])

print(f"\nLondon_ON: {on_train_len} train, {on_val_len} val")
print(f"London_UK: {uk_train_len} train, {uk_val_len} val")
print(f"Total train: {len(train_dataset)}, Total val: {len(val_dataset)}")

import shutil
import random
from pathlib import Path

source_dir = Path("data")
target_dir = Path("report_data")
train_ratio = 0.7

# Clean old directories if they exist
if (target_dir / "train").exists():
    shutil.rmtree(target_dir / "train")
if (target_dir / "val").exists():
    shutil.rmtree(target_dir / "val")

# Make new folders
for split in ["train", "val"]:
    for city in ["London_ON", "London_UK"]:
        (target_dir / split / city).mkdir(parents=True, exist_ok=True)

# Helper to copy images
def copy_images(images, split, city):
    for img in images:
        shutil.copy(img, target_dir / split / city / img.name)

# Process each city
for city_src, city_dest in [("london_on", "London_ON"), ("london_uk", "London_UK")]:
    all_images = []
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
    copy_images(train_imgs, "train", city_dest)
    copy_images(val_imgs, "val", city_dest)
    print(f"{city_dest}: {len(train_imgs)} train, {len(val_imgs)} val (of {len(all_images)})")

print("Physical split complete. Check your 'report_data' folder.")
