import time
from pathlib import Path
from config import (
    DATA_DIR, METADATA_CSV_PATH, GOOGLE_MAPS_API_KEY
)
from sample_coords import sample_nearby_coordinates
from get_pano_data import get_all_pano_data
from download_panorama import download_panorama
from process_image import crop_and_resize, save_processed_image
from classify_season import classify_season
from classify_blur import classify_sharpness
from log_metadata import log_metadata

# Collect 20,000 total → 1250 per (city, season, sharpness)
TARGET_PER_CATEGORY = 1250
MAX_ATTEMPTS = 10000  # Per category

def collect_images_for_category(city, season, sharpness, target_n=TARGET_PER_CATEGORY):
    count = 0
    attempts = 0

    while count < target_n and attempts < MAX_ATTEMPTS:
        attempts += 1
        coords_list = sample_nearby_coordinates(city, n=3)
        for lat, lon in coords_list:
            pano_list = get_all_pano_data(lat, lon, api_key=GOOGLE_MAPS_API_KEY)
            for pano in pano_list:
                if classify_season(pano["date"]) != season:
                    continue

                pano_id = pano["pano_id"]
                raw_image = download_panorama(pano_id, city, season, verbose=False)
                if raw_image is None:
                    continue

                processed = crop_and_resize(raw_image)
                sharp_label = classify_sharpness(processed)
                if sharp_label != sharpness:
                    continue

                filename = f"{city}_{pano['date']}_{pano_id}.jpg"
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
    for city in ["london_uk", "london_on"]:
        for season in ["winter", "spring", "summer", "fall"]:
            for sharpness in ["sharp", "blurry"]:
                print(f"\n🔄 Collecting: {city} / {season} / {sharpness}")
                collect_images_for_category(city, season, sharpness)

if __name__ == "__main__":
    main()