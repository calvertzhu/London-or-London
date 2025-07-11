import os
from pathlib import Path
from dotenv import load_dotenv

# Load API Key
env_path = Path.home() / "Downloads" / "google_maps_API.env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    raise FileNotFoundError(f"API key file not found at: {env_path}")

GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
if not GOOGLE_MAPS_API_KEY:
    raise ValueError("Missing GOOGLE_MAPS_API_KEY in .env file.")

# Project Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
METADATA_DIR = PROJECT_ROOT / "metadata"
METADATA_CSV_PATH = METADATA_DIR / "combined.csv"

# Image save path pattern (used if needed)
IMAGE_SAVE_PATH = DATA_DIR / "{city}" / "{season}" / "{sharpness}"

# Bounding Boxes for Each City
CITY_BOUNDING_BOXES = {
    "london_uk": {
        "lat_min": 51.25,   # south (closer to Sutton / Croydon)
        "lat_max": 51.72,   # north (up to Waltham Cross)
        "lon_min": -0.55,   # west (Egham, Heathrow)
        "lon_max": 0.25,    # east (Romford edge)
    },
    "london_on": {
        "lat_min": 42.90,   # includes more of southern outskirts
        "lat_max": 43.10,   # includes Masonville, edge of Arva
        "lon_min": -81.45,  # more west toward Komoka
        "lon_max": -81.10,  # east edge of London, past the airport
    },
}

# Image Processing
CROP_SIZE = 224               # Final square crop size
JPEG_QUALITY = 94             # JPEG compression quality
BLUR_THRESHOLD = 20.0        # Laplacian variance threshold

# Data Collection
TARGET_TOTAL_IMAGES = 20000   # Used in full dataset run
MIN_SAMPLE_DISTANCE_METERS = 50
API_TIMEOUT_SECONDS = 5

SEASON_MAPPING = {
    "winter": [12, 1, 2],
    "spring": [3, 4, 5],
    "summer": [6, 7, 8],
    "fall": [9, 10, 11],
}
