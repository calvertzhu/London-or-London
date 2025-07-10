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

TARGET_TOTAL = 5
MAX_ATTEMPTS = 100000

def collect():
    count = 0
    attempts = 0

    while count < TARGET_TOTAL and attempts < MAX_ATTEMPTS:
        for city in ["london_on"]:
            coords_list = sample_nearby_coordinates(city, n=3)
            for lat, lon in coords_list:
                print(f"\n📍 Sampling {city} at ({lat:.5f}, {lon:.5f})")
                pano_list = get_all_pano_data(lat, lon)

                if not pano_list:
                    print(f"⚠️ No panoramas found at this location.")
                    continue

                print(f"📸 Found {len(pano_list)} panoramas.")

                for pano in pano_list:
                    pano_id = pano["pano_id"]
                    date = pano["date"]
                    print(f"🔍 Processing pano {pano_id}, date: {date}")

                    raw_image = download_panorama(pano_id, verbose=True)
                    if raw_image is None:
                        print(f"❌ Skipping {pano_id} (download failed)")
                        continue

                    processed = crop_and_resize(raw_image)
                    season = classify_season(date)
                    sharpness = classify_sharpness(processed)
                    print(f"✅ Classifications → season: {season}, sharpness: {sharpness}")

                    filename = f"{pano_id}_{city}_{season}_{sharpness}.jpg"

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
                        print("\n✅ Collection complete.")
                        return

                time.sleep(0.1)

        attempts += 1

    print(f"\n⚠️ Only collected {count} images after {MAX_ATTEMPTS} attempts.")

if __name__ == "__main__":
    collect()
