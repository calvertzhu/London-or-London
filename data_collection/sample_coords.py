# sample_coords.py

import random
from config import CITY_BOUNDING_BOXES

def sample_coordinate(city_name):
    """
    Randomly sample a lat/lon within the bounding box of a given city.

    Args:
        city_name (str): 'london_uk' or 'london_on'

    Returns:
        tuple: (lat, lon)
    """
    box = CITY_BOUNDING_BOXES[city_name]
    lat = random.uniform(box["lat_min"], box["lat_max"])
    lon = random.uniform(box["lon_min"], box["lon_max"])
    return lat, lon

# Example usage
if __name__ == "__main__":
    for city in ["london_uk", "london_on"]:
        for _ in range(3):
            lat, lon = sample_coordinate(city)
            print(f"{city.upper()} → lat: {lat:.6f}, lon: {lon:.6f}")
