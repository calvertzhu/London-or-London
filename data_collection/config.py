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
        "lat_min": 51.28,
        "lat_max": 51.70,
        "lon_min": -0.51,
        "lon_max": 0.33,
    },
    "london_on": {
        "lat_min": 42.85,
        "lat_max": 43.10,
        "lon_min": -81.40,
        "lon_max": -81.15,
    },
}

# Image Processing
CROP_SIZE = 224               # Final square crop size
JPEG_QUALITY = 100             # JPEG compression quality
BLUR_THRESHOLD = 50.0        # Laplacian variance threshold

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
