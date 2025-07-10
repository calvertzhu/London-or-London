import time
from pathlib import Path
from collections import defaultdict
from data_collection.config import DATA_DIR, METADATA_CSV_PATH, GOOGLE_MAPS_API_KEY
from data_collection.sample_coords import sample_nearby_coordinates
from data_collection.get_pano_data import get_all_pano_data
from data_collection.download_panorama import download_panorama
from data_collection.process_image import crop_and_resize, save_processed_image
from data_collection.classify_season import classify_season
from data_collection.classify_blur import classify_sharpness
from data_collection.log_metadata import log_metadata
from data_collection.explore_metadata import analyze_metadata

# Settings
TARGET_PER_CATEGORY = 5
MAX_ATTEMPTS = 10000

def test_random_small():
    category_counts = defaultdict(int)
    categories = [
        (city, season, sharpness)
        for city in ["london_uk", "london_on"]
        for season in ["winter", "spring", "summer", "fall"]
        for sharpness in ["sharp", "blurry"]
    ]

    attempts = 0
    while attempts < MAX_ATTEMPTS:
        target_category = min(categories, key=lambda c: category_counts[c])
        if all(category_counts[c] >= TARGET_PER_CATEGORY for c in categories):
            break

        city, season_target, sharpness_target = target_category
        coords_list = sample_nearby_coordinates(city, n=2)
        for lat, lon in coords_list:
            pano_list = get_all_pano_data(lat, lon, api_key=GOOGLE_MAPS_API_KEY)
            for pano in pano_list:
                pano_id = pano["pano_id"]
                date = pano["date"]

                raw_image = download_panorama(pano_id, verbose=False)
                if raw_image is None:
                    continue

                processed = crop_and_resize(raw_image)
                season_actual = classify_season(date)
                sharpness_actual = classify_sharpness(processed)

                if season_actual != season_target or sharpness_actual != sharpness_target:
                    continue

                filename = f"TEST_{pano_id}_{city}_{season_actual}_{sharpness_actual}.jpg"

                save_processed_image(
                    image=processed,
                    city=city,
                    season=season_actual,
                    sharpness=sharpness_actual,
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
                    "season": season_actual,
                    "sharpness": sharpness_actual,
                    "pano_id": pano_id
                }, METADATA_CSV_PATH)

                category_counts[(city, season_actual, sharpness_actual)] += 1
                print(f"✅ [{category_counts[(city, season_actual, sharpness_actual)]}/{TARGET_PER_CATEGORY}] {city}/{season_actual}/{sharpness_actual} → {filename}")

                if all(category_counts[c] >= TARGET_PER_CATEGORY for c in categories):
                    break

            time.sleep(0.1)
        attempts += 1

    print("\n📊 Final sample counts:")
    for c in categories:
        print(f"{c[0]:<10} | {c[1]:<6} | {c[2]:<7} → {category_counts[c]}")

    print("\n📈 Running metadata analysis...")
    analyze_metadata()

if __name__ == "__main__":
    test_random_small()
