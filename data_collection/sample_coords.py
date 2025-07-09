import random
from data_collection.config import CITY_BOUNDING_BOXES

def sample_nearby_coordinates(city, center_lat=None, center_lon=None, n=5, radius_deg=0.001):
    """
    Sample 'n' GPS coordinates around a center point in a city.
    If center is not provided, randomly pick one from the bounding box.

    Args:
        city (str): 'london_uk' or 'london_on'
        center_lat (float): Optional center latitude
        center_lon (float): Optional center longitude
        n (int): Number of points to sample
        radius_deg (float): Perturbation radius in degrees

    Returns:
        list of (lat, lon) tuples
    """
    box = CITY_BOUNDING_BOXES[city]

    if center_lat is None or center_lon is None:
        center_lat = random.uniform(box["lat_min"], box["lat_max"])
        center_lon = random.uniform(box["lon_min"], box["lon_max"])

    coords = []
    for _ in range(n):
        lat_offset = random.uniform(-radius_deg, radius_deg)
        lon_offset = random.uniform(-radius_deg, radius_deg)
        lat = center_lat + lat_offset
        lon = center_lon + lon_offset
        coords.append((lat, lon))

    return coords