def generate_filename(pano_id, lat, lon, date, season, sharpness, suffix=""):
    """
    Generate a standardized filename for a processed image.

    Example output:
    Xyz123_43p12345_-81p12345_202306_summer_sharp.jpg

    Args:
        pano_id (str): Google panorama ID
        lat (float): Latitude
        lon (float): Longitude
        date (str): YYYY-MM
        season (str): 'spring', 'fall', etc.
        sharpness (str): 'sharp' or 'blurry'
        suffix (str): Optional suffix (e.g. '_aug1')

    Returns:
        str: filename
    """
    lat_str = f"{lat:.5f}".replace(".", "p")
    lon_str = f"{lon:.5f}".replace(".", "p")
    date_str = date.replace("-", "")
    return f"{pano_id}_{lat_str}_{lon_str}_{date_str}_{season}_{sharpness}{suffix}.jpg"
