import time
from data_collection.config import DATA_DIR, METADATA_CSV_PATH
from data_collection.sample_coords import sample_nearby_coordinates
from data_collection.get_pano_data import get_all_pano_data
from data_collection.download_panorama import download_panorama
from data_collection.process_image import crop_and_resize, save_processed_image
from data_collection.classify_season import classify_season
from data_collection.classify_blur import classify_sharpness
from data_collection.log_metadata import log_metadata
from data_collection.utils import generate_filename

# Default values
DEFAULT_TARGET_TOTAL = 40
MAX_ATTEMPTS = 100000

def collect(city, target_total=DEFAULT_TARGET_TOTAL, run_id=None):
    count = 0
    attempts = 0

    while count < target_total and attempts < MAX_ATTEMPTS:
        coords_list = sample_nearby_coordinates(city, n=3)

        for lat, lon in coords_list:
            print(f"\nSampling {city} at ({lat:.5f}, {lon:.5f})")
            pano_list = get_all_pano_data(lat, lon)

            if not pano_list:
                print("No panoramas found.")
                continue

            for pano in pano_list:
                pano_id = pano["pano_id"]
                date = pano["date"]
                print(f"Processing pano {pano_id}, date: {date}")

                raw_image = download_panorama(pano_id, verbose=True)
                if raw_image is None:
                    print("Download failed.")
                    continue

                # Classify before resizing
                season = classify_season(date)
                sharpness = classify_sharpness(raw_image)

                # Now resize/crop after classification
                processed = crop_and_resize(raw_image, position="center")
                print(f"→ season: {season}, sharpness: {sharpness}")

                # Build filename consistent with save function
                filename = generate_filename(
                    pano_id, pano["lat"], pano["lon"], date, season, sharpness
                )       
                
                # Save image
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

                # Log metadata
                metadata = {
                    "filename": filename,
                    "city": city,
                    "lat": pano["lat"],
                    "lon": pano["lon"],
                    "date": date,
                    "season": season,
                    "sharpness": sharpness,
                    "pano_id": pano_id,
                    "source": "original"
                }
                if run_id:
                    metadata["run_id"] = run_id

                log_metadata(metadata, METADATA_CSV_PATH)

                count += 1
                print(f"[{count}/{target_total}] Saved {filename}")

                if count >= target_total:
                    print("\nCollection complete.")
                    return

            time.sleep(0.1)

        attempts += 1

    print(f"\nCollected {count} images after {MAX_ATTEMPTS} attempts.")

if __name__ == "__main__":
    collect(city="london_on", target_total=40)
