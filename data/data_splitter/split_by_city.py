import os
import csv
import random
import shutil

def load_split_log(log_path):
    if not os.path.exists(log_path):
        return set()
    with open(log_path, newline='') as f:
        reader = csv.DictReader(f)
        return set(row['filepath'] for row in reader)

def save_to_log(log_path, new_entries):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    fieldnames = ['filepath', 'split', 'city', 'season', 'original_filename', 'new_filename']
    file_exists = os.path.exists(log_path)
    with open(log_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for entry in new_entries:
            writer.writerow(entry)

def collect_images(dir_path, city, season):
    if not os.path.isdir(dir_path):
        print(f"Warning: Directory not found: {dir_path}")
        return []
    images = []
    for fname in os.listdir(dir_path):
        if fname.lower().endswith(".jpg"):
            full_path = os.path.join(dir_path, fname)
            images.append((full_path, city, season, fname))
    return images

def split_and_copy(images, split_ratios, dest_root, log_path):
    random.shuffle(images)
    n_total = len(images)
    n_train = int(split_ratios[0] * n_total)
    n_val = int(split_ratios[1] * n_total)
    n_test = n_total - n_train - n_val

    splits = ['train'] * n_train + ['val'] * n_val + ['test'] * n_test
    new_log_entries = []
    existing_files = load_split_log(log_path)

    for (img_path, city, season, orig_name), split in zip(images, splits):
        if img_path in existing_files:
            continue

        ext = os.path.splitext(orig_name)[1]
        new_name = f"{city}_{season}_{random.randint(100000, 999999)}{ext}"
        dest_dir = os.path.join(dest_root, split, city)
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, new_name)

        shutil.copyfile(img_path, dest_path)

        new_log_entries.append({
            'filepath': img_path,
            'split': split,
            'city': city,
            'season': season,
            'original_filename': orig_name,
            'new_filename': new_name
        })

    save_to_log(log_path, new_log_entries)

# Configuration

logPath = "data/split_log.csv"
destRoot = "data/split_data"
splitRatios = (0.7, 0.15, 0.15)

# The file paths dont read for some reason, need fix :(
londonOnDirs = [("data/london_on/summer", "london_on", "summer")]

londonUkDirs = [
    ("data/london_uk/fall/sharp", "london_uk", "fall"),
    ("data/london_uk/summer/sharp", "london_uk", "summer"),
    ("data/london_uk/winter/sharp", "london_uk", "winter")
]

allImages = []
for path, city, season in londonOnDirs + londonUkDirs:
    allImages.extend(collect_images(path, city, season))

split_and_copy(allImages, splitRatios, destRoot, logPath)
print("Done splitting and copying.")
