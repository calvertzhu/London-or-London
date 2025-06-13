import os
import random
from pathlib import Path
from streetview import search_panoramas, get_panorama

# Load API key
API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
if API_KEY is None:
    raise ValueError("Missing API key. Set GOOGLE_MAPS_API_KEY in your environment.")

# Small bounding box around central London
lat_min, lat_max = 51.500, 51.505
lon_min, lon_max = -0.130, -0.120

# Generate a random coordinate in the box
lat = random.uniform(lat_min, lat_max)
lon = random.uniform(lon_min, lon_max)

# Search for a nearby panorama
panos = search_panoramas(lat=lat, lon=lon)
if not panos:
    raise RuntimeError("No panoramas found near the random coordinate.")

# Select the first panorama
pano_id = panos[0].pano_id

# Download the full panorama
image = get_panorama(pano_id=pano_id)

# Save to Downloads
save_dir = Path.home() / "Downloads"
save_dir.mkdir(parents=True, exist_ok=True)
filename = save_dir / f"panorama_london_uk_{lat:.5f}_{lon:.5f}.jpg"
image.save(str(filename), "jpeg")

print(f"Saved panoramic image to: {filename}")
