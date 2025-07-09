from streetview import search_panoramas, get_panorama_meta
from data_collection.config import GOOGLE_MAPS_API_KEY
import time

def get_all_pano_data(lat, lon, max_panos=10, verbose=True):
    """
    Retrieve all available panoramas near a location with pano_id, lat/lon, and date.
    
    Args:
        lat (float): Latitude
        lon (float): Longitude
        max_panos (int): Max number of pano results to check
        verbose (bool): Whether to print progress
    
    Returns:
        list[dict]: List of pano metadata:
            {
                'pano_id': str,
                'lat': float,
                'lon': float,
                'date': 'YYYY-MM'
            }
    """
    try:
        panos = search_panoramas(lat=lat, lon=lon)
    except Exception as e:
        if verbose:
            print(f"❌ Failed pano search at ({lat}, {lon}): {e}")
        return []

    pano_results = []
    seen_ids = set()

    for pano in panos[:max_panos]:
        if pano.pano_id in seen_ids:
            continue
        seen_ids.add(pano.pano_id)

        try:
            meta = get_panorama_meta(pano_id=pano.pano_id, api_key=GOOGLE_MAPS_API_KEY)
            if meta and meta.date:
                pano_results.append({
                    "pano_id": pano.pano_id,
                    "lat": meta.location.lat,
                    "lon": meta.location.lng,
                    "date": meta.date
                })
        except Exception as e:
            if verbose:
                print(f"⚠️ Metadata error for {pano.pano_id}: {e}")
        time.sleep(0.1)  # Avoid hitting rate limit

    return pano_results
