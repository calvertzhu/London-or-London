# data_collection/test_collect.py

import time
import sys
from pathlib import Path

from data_collection.config import DATA_DIR, METADATA_CSV_PATH
from data_collection.sample_coords import sample_nearby_coordinates
from data_collection.get_pano_data import get_all_pano_data
from data_collection.download_panorama import download_panorama
from data_collection.process_image import crop_and_resize, save_processed_image
from data_collection.classify_season import classify_season
from data_collection.classify_blur import classify_sharpness
from data_collection.log_metadata import log_metadata


# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Target: 5 categories × 20 images = 100 total
TARGET_PER_CATEGORY = 20
MAX_ATTEMPTS = 1000

def collect_images_for_category(city, season, sharpness, target_n=TARGET_PER_CATEGORY):
    count = 0
    attempts = 0

    while count < target_n and attempts < MAX_ATTEMPTS:
        attempts += 1
        coords_list = sample_nearby_coordinates(city, n=3)
        for lat, lon in coords_list:
            pano_list = get_all_pano_data(lat, lon)
            for pano in pano_list:
                if classify_season(pano["date"]) != season:
                    continue

                pano_id = pano["pano_id"]
                filename = f"{city}_{pano['date']}_{pano_id}.jpg"

                # Skip if image already exists
                target_path = DATA_DIR / city / season / sharpness / filename
                if target_path.exists():
                    continue

                raw_image = download_panorama(pano_id, city, season, verbose=False)
                if raw_image is None:
                    continue

                processed = crop_and_resize(raw_image)
                sharp_label = classify_sharpness(processed)
                if sharp_label != sharpness:
                    continue

                save_processed_image(processed, city, season, sharp_label, pano_id, DATA_DIR)

                log_metadata({
                    "filename": filename,
                    "city": city,
                    "lat": pano["lat"],
                    "lon": pano["lon"],
                    "date": pano["date"],
                    "season": season,
                    "sharpness": sharp_label,
                    "pano_id": pano_id
                }, METADATA_CSV_PATH)

                count += 1
                print(f"✅ [{count}/{target_n}] Collected: {filename}")
                if count >= target_n:
                    return
            time.sleep(0.1)

    if count < target_n:
        print(f"⚠️ Not enough samples for {city} / {season} / {sharpness} (got {count})")

def main():
    test_categories = [
        ("london_uk", "summer", "sharp"),
        ("london_uk", "summer", "blurry"),
        ("london_on", "winter", "sharp"),
        ("london_on", "winter", "blurry"),
        ("london_on", "fall", "sharp"),
    ]

    for city, season, sharpness in test_categories:
        print(f"\n🔄 Collecting: {city} / {season} / {sharpness}")
        collect_images_for_category(city, season, sharpness)

if __name__ == "__main__":
    main()
