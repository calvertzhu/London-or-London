import csv
from pathlib import Path

def log_metadata(entry: dict, csv_path: Path):
    """
    Append a single row of metadata to a CSV file. Creates header if file is new.

    Args:
        entry (dict): Metadata row with keys:
            ['filename', 'city', 'lat', 'lon', 'date', 'season', 'sharpness', 'pano_id', 'run_id', 'source']
        csv_path (Path): Path to metadata CSV file
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()

    # Updated fieldnames to include all known metadata keys
    fieldnames = [
        "filename", "city", "lat", "lon", "date",
        "season", "sharpness", "pano_id", "run_id", "source"
    ]

    # Fill missing optional fields with empty string if necessary
    for key in fieldnames:
        entry.setdefault(key, "")

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(entry)
