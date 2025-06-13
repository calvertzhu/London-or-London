import os
import random
from pathlib import Path
from streetview import search_panoramas, get_panorama

# Load API key
API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
if API_KEY is None:
    raise ValueError("Missing API key. Set GOOGLE_MAPS_API_KEY in your environment.")

# Bounding box
lat_min, lat_max = 51.500, 51.505
lon_min, lon_max = -0.130, -0.120

# Save directory
save_dir = Path.home() / "Downloads" / "panorama_london_uk"
save_dir.mkdir(parents=True, exist_ok=True)

# Loop to get 5 images
downloaded = 0 
attempts = 0 
max_attempts = 20  # avoid infinite loops 

while downloaded < 5 and attempts < max_attempts:
    attempts += 1

    # Random coordinate
    lat = random.uniform(lat_min, lat_max)
    lon = random.uniform(lon_min, lon_max)

    try:
        # Search for panorama
        panos = search_panoramas(lat=lat, lon=lon)
        if not panos:
            continue 

        pano_id = panos[0].pano_id

        # Get full panorama image
        image = get_panorama(pano_id=pano_id)

        # Resize and compress
        resized_size = (image.width // 4, image.height // 4)
        image = image.resize(resized_size)

        # Save image
        filename = save_dir / f"london_uk_pano_{downloaded+1}_{lat:.5f}_{lon:.5f}.jpg"
        image.save(str(filename), format="JPEG", quality=40)

        print(f"[{downloaded+1}/5] Saved to: {filename}")
        downloaded += 1

    except Exception as e:
        print(f"Skipped location due to error: {e}")
        continue
