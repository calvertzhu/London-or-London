# data_collection/config.py

import os
from pathlib import Path
from dotenv import load_dotenv

# ─────────────────────────────────────────────
# 1. Load API key from .env file in Downloads
# ─────────────────────────────────────────────

env_path = Path.home() / "Downloads" / "google_maps_API.env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    raise FileNotFoundError(f"API key file not found at: {env_path}")

GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
if not GOOGLE_MAPS_API_KEY:
    raise ValueError("Missing GOOGLE_MAPS_API_KEY in .env file.")

# ─────────────────────────────────────────────
# 2. Project Paths
# ─────────────────────────────────────────────

# Base project root (e.g., ~/London-or-London/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data and metadata folders
DATA_DIR = PROJECT_ROOT / "data"
METADATA_DIR = PROJECT_ROOT / "metadata"

# Where to save cropped/resized images
IMAGE_SAVE_PATH = DATA_DIR / "{city}" / "{season}" / "{sharpness}"

# Path to log metadata CSV
METADATA_CSV_PATH = METADATA_DIR / "combined.csv"

# ─────────────────────────────────────────────
# 3. Bounding Boxes for Each City
# ─────────────────────────────────────────────

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

# ─────────────────────────────────────────────
# 4. Image Processing Config
# ─────────────────────────────────────────────

CROP_SIZE = 224             # Final image size (square crop)
JPEG_QUALITY = 60           # JPEG compression quality (1–100)
BLUR_THRESHOLD = 100.0      # Laplacian variance threshold for sharp/blurry

# ─────────────────────────────────────────────
# 5. Data Collection Config
# ─────────────────────────────────────────────

TARGET_IMAGES_PER_CATEGORY = 1250      # 1250 per (city, season, sharpness)
MIN_SAMPLE_DISTANCE_METERS = 50        # Prevent redundant images
API_TIMEOUT_SECONDS = 5                # Request timeout (if used)

# Define month-to-season mapping
SEASON_MAPPING = {
    "winter": [12, 1, 2],
    "spring": [3, 4, 5],
    "summer": [6, 7, 8],
    "fall": [9, 10, 11],
}
