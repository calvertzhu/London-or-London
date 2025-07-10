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
    fieldnames = ['filepath', 'split', 'city', 'season', 'original_filename', 'new_filename']
    file_exists = os.path.exists(log_path)
    with open(log_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(new_entries)

def collect_images(base_path, city, season=None):
    """Collect full and relative paths, and attach city/season info."""
    results = []
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith('.png'):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, start='data')
                results.append({
                    'full_path': full_path,
                    'rel_path': rel_path,
                    'city': city,
                    'season': season if season else 'unknown',
                    'original_filename': file
                })
    return results

def generate_unique_filename(entry):
    base = os.path.splitext(entry['original_filename'])[0]
    city = entry['city']
    season = entry['season']
    return f"{base}_{city}_{season}.png"

def split_dataset_incremental(source_dir, output_dir, log_path, split_ratio=(0.7, 0.15, 0.15), seed=42):
    random.seed(seed)
    assert sum(split_ratio) == 1.0

    prev_files = load_split_log(log_path)
    new_log_entries = []

    for split in ['train', 'val', 'test']:
        for city in ['london_on', 'london_uk']:
            os.makedirs(os.path.join(output_dir, split, city), exist_ok=True)

    # ON images
    on_images = collect_images(os.path.join(source_dir, 'london_on', 'summer'), city='london_on', season='summer')

    # UK images from multiple seasons
    uk_images = []
    for season in ['fall', 'summer', 'winter']:
        uk_images += collect_images(
            os.path.join(source_dir, 'london_uk', season, 'sharp'),
            city='london_uk',
            season=season
        )

    all_images = on_images + uk_images
    new_images = [img for img in all_images if img['rel_path'] not in prev_files]
    random.shuffle(new_images)

    n_total = len(new_images)
    n_train = int(split_ratio[0] * n_total)
    n_val = int(split_ratio[1] * n_total)

    split_map = {
        'train': new_images[:n_train],
        'val': new_images[n_train:n_train + n_val],
        'test': new_images[n_train + n_val:]
    }

    for split, images in split_map.items():
        for entry in images:
            unique_name = generate_unique_filename(entry)
            dst_path = os.path.join(output_dir, split, entry['city'], unique_name)
            shutil.copy(entry['full_path'], dst_path)

            new_log_entries.append({
                'filepath': entry['rel_path'],
                'split': split,
                'city': entry['city'],
                'season': entry['season'],
                'original_filename': entry['original_filename'],
                'new_filename': unique_name
            })

    save_to_log(log_path, new_log_entries)
    print(f"Added {len(new_log_entries)} new images with unique names and season tracking.")

if __name__ == "__main__":
    split_dataset_incremental(
        source_dir="data",
        output_dir="split_data",
        log_path="split_data/split_log.csv"
    )

