from streetview import get_panorama
from PIL import Image
import io

def download_panorama(pano_id, verbose=True):
    """
    Download a stitched panorama image given a pano_id, does not save the raw image to disk.
    
    Args:
        pano_id (str): Google panorama ID
        verbose (bool): Whether to print log messages

    Returns:
        PIL.Image.Image or None if failed
    """
    try:
        image = get_panorama(pano_id=pano_id)
        if image is None or not isinstance(image, Image.Image):
            if verbose:
                print(f"Failed to download valid image for {pano_id}")
            return None
        if verbose:
            print(f"Downloaded pano {pano_id} successfully")
        return image

    except Exception as e:
        if verbose:
            print(f"Error downloading pano {pano_id}: {e}")
        return None
