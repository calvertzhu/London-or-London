import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")
OUTPUT_CSV = Path("metadata/regenerated_metadata.csv")

def parse_filename(filename):
    """
    Extract metadata fields from a standardized filename:
    {pano_id}_{lat}_{lon}_{date}_{season}_{sharpness}.jpg
    """
    name = Path(filename).stem
    parts = name.split("_")

    if len(parts) < 6:
        raise ValueError(f"Unexpected filename format: {filename}")

    sharpness = parts[-1]
    season = parts[-2]
    date_str = parts[-3]
    lon_str = parts[-4]
    lat_str = parts[-5]
    pano_id = "_".join(parts[:-5])

    try:
        lat = float(lat_str.replace("p", "."))
        lon = float(lon_str.replace("p", "."))
    except ValueError:
        raise ValueError(f"Could not parse lat/lon in: {filename}")

    date = f"{date_str[:4]}-{date_str[4:]}"  # convert 201804 → 2018-04

    return {
        "filename": filename,
        "pano_id": pano_id,
        "lat": lat,
        "lon": lon,
        "date": date,
        "season": season,
        "sharpness": sharpness
    }

def regenerate_metadata():
    rows = []

    for image_path in DATA_DIR.rglob("*.jpg"):
        try:
            parts = image_path.parts
            if len(parts) < 4:
                continue
            city = parts[-4]

            meta = parse_filename(image_path.name)
            meta["city"] = city
            rows.append(meta)

        except Exception as e:
            print(f"Failed to parse {image_path.name}: {e}")

    df = pd.DataFrame(rows)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nRegenerated metadata to: {OUTPUT_CSV}")
    print(f"Total rows: {len(df)}")

if __name__ == "__main__":
    regenerate_metadata()
