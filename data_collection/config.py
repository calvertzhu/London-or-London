import os
from pathlib import Path
from dotenv import load_dotenv

### API Keys and Paths

env_path = Path.home() / "Downloads" / "google_maps_API.env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    raise FileNotFoundError(f"API key file not found at: {env_path}")

GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
if GOOGLE_MAPS_API_KEY is None:
    raise ValueError("Missing GOOGLE_MAPS_API_KEY")

# Base directory paths (relative to project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
METADATA_DIR = PROJECT_ROOT / "metadata"

# Image storage path format
IMAGE_SAVE_PATH = DATA_DIR / "{city}" / "{season}" / "{sharpness}"

# Metadata log file
METADATA_CSV_PATH = METADATA_DIR / "combined.csv"

### Bounding Boxes (lat_min, lat_max, lon_min, lon_max)

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

### Image Processing

CROP_SIZE = 224  # Final image size (square)
JPEG_QUALITY = 60  # Compression level

# Laplacian threshold to classify sharp vs. blurry
BLUR_THRESHOLD = 100.0

### xData Collection

# Number of images to collect per (city, season, sharpness)
TARGET_IMAGES_PER_CATEGORY = 500

# Minimum distance (meters) between samples to avoid duplicates
MIN_SAMPLE_DISTANCE_METERS = 50

# Request timeout or retry settings
API_TIMEOUT_SECONDS = 5

# Allowed months per season
SEASON_MAPPING = {
    "winter": [12, 1, 2],
    "spring": [3, 4, 5],
    "summer": [6, 7, 8],
    "fall": [9, 10, 11],
}
