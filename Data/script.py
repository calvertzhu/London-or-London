import os
from pathlib import Path
from streetview import search_panoramas, get_streetview

# Step 1: Load your API key
API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
if API_KEY is None:
    raise ValueError("Missing API key. Set GOOGLE_MAPS_API_KEY in your environment.")

# Step 2: Choose coordinates (London, UK — Big Ben area)
lat = 51.5007
lon = -0.1246

# Step 3: Search for available panoramas
panos = search_panoramas(lat=lat, lon=lon)
if not panos:
    raise RuntimeError("No panoramas found for these coordinates.")

# Step 4: Select the first panorama
pano_id = panos[0].pano_id

# Step 5: Download the Street View image (640x640 facing default heading)
image = get_streetview(pano_id=pano_id, api_key=API_KEY)

# Step 6: Save the image to Downloads
save_dir = Path.home() / "Downloads"
save_dir.mkdir(parents=True, exist_ok=True)
filename = save_dir / f"london_uk_{lat}_{lon}.jpg"
image.save(str(filename), "jpeg")

print(f"✅ Street View image saved to: {filename}")
