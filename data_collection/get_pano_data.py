from streetview import search_panoramas, get_panorama_meta
from config import GOOGLE_MAPS_API_KEY

def get_valid_pano_metadata(lat, lon):
    """
    Search for panoramas near the given GPS coordinates and return metadata
    for the first available panorama.

    Args:
        lat (float): Latitude
        lon (float): Longitude

    Returns:
        dict: {
            'pano_id': str,
            'lat': float,
            'lon': float,
            'date': 'YYYY-MM'
        } or None if no pano found
    """
    panos = search_panoramas(lat=lat, lon=lon)

    if not panos:
        return None  # No coverage here

    pano_id = panos[0].pano_id
    meta = get_panorama_meta(pano_id=pano_id, api_key=GOOGLE_MAPS_API_KEY)

    if meta is None or meta.date is None:
        return None  # Metadata retrieval failed

    return {
        "pano_id": pano_id,
        "lat": meta.location.lat,
        "lon": meta.location.lng,
        "date": meta.date 
    }