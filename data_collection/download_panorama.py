from streetview import get_panorama
from pathlib import Path
from data_collection.config import DATA_DIR

def download_panorama(pano_id, city, season, save_dir=None, filename=None, verbose=True):
    """
    Download a stitched panorama image given a pano_id and save it to the data directory.

    Args:
        pano_id (str): Google panorama ID
        city (str): 'london_uk' or 'london_on'
        season (str): 'winter', 'spring', 'summer', 'fall'
        save_dir (Path): Optional override for save directory
        filename (str): Optional override for filename
        verbose (bool): Print log messages

    Returns:
        PIL.Image.Image or None if failed
    """

    try:
        # Download stitched pano from API
        image = get_panorama(pano_id=pano_id)
        if image is None:
            if verbose:
                print(f"Failed to download image for {pano_id}")
            return None
    except Exception as e:
        if verbose:
            print(f"Error downloading pano {pano_id}: {e}")
        return None

    # Build file save path
    if save_dir is None:
        save_dir = DATA_DIR / city / season / "raw"
    save_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = f"{pano_id}.jpg"
    save_path = save_dir / filename

    # Save image
    try:
        image.save(save_path, "jpeg")
        if verbose:
            print(f"Saved panorama to {save_path}")
    except Exception as e:
        if verbose:
            print(f"Error saving image to {save_path}: {e}")
        return None

    return image