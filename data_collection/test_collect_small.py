import time
from pathlib import Path
from data_collection.config import DATA_DIR, METADATA_CSV_PATH
from data_collection.sample_coords import sample_nearby_coordinates
from data_collection.get_pano_data import get_all_pano_data
from data_collection.download_panorama import download_panorama
from data_collection.process_image import crop_and_resize, save_processed_image
from data_collection.classify_season import classify_season
from data_collection.classify_blur import classify_sharpness
from data_collection.log_metadata import log_metadata
from data_collection.explore_metadata import analyze_metadata
from data_collection.balance_dataset import balance_dataset

TARGET_TOTAL = 100  # total images to collect (unfiltered)
MAX_ATTEMPTS = 10000

def test_collect_fast():
    count = 0
    attempts = 0

    while count < TARGET_TOTAL and attempts < MAX_ATTEMPTS:
        for city in ["london_uk", "london_on"]:
            coords_list = sample_nearby_coordinates(city, n=2)
            for lat, lon in coords_list:
                pano_list = get_all_pano_data(lat, lon)
                for pano in pano_list:
                    pano_id = pano["pano_id"]
                    date = pano["date"]

                    raw_image = download_panorama(pano_id, verbose=False)
                    if raw_image is None:
                        continue

                    processed = crop_and_resize(raw_image)
                    season = classify_season(date)
                    sharpness = classify_sharpness(processed)

                    filename = f"TEST_{pano_id}_{city}_{season}_{sharpness}.jpg"

                    save_processed_image(
                        image=processed,
                        city=city,
                        season=season,
                        sharpness=sharpness,
                        pano_id=pano_id,
                        lat=pano["lat"],
                        lon=pano["lon"],
                        date=date,
                        save_dir=DATA_DIR
                    )

                    log_metadata({
                        "filename": filename,
                        "city": city,
                        "lat": pano["lat"],
                        "lon": pano["lon"],
                        "date": date,
                        "season": season,
                        "sharpness": sharpness,
                        "pano_id": pano_id
                    }, METADATA_CSV_PATH)

                    count += 1
                    print(f"✅ [{count}/{TARGET_TOTAL}] {city}/{season}/{sharpness} → {filename}")
                    if count >= TARGET_TOTAL:
                        break

                time.sleep(0.1)
        attempts += 1

    print(f"\n📦 Collected {count} images.")
    print("📈 Running metadata analysis...")
    analyze_metadata()

    print("🎯 Balancing to 2 per (city, season, sharpness)...")
    balance_dataset(n_per_class=2)

if __name__ == "__main__":
    test_collect_fast()