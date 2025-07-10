from PIL import Image
from pathlib import Path
from data_collection.config import CROP_SIZE, JPEG_QUALITY

def crop_and_resize(image, size=CROP_SIZE):
    """
    Crop the center square of a panorama and resize to target size.

    Args:
        image (PIL.Image): Full panoramic image (typically very wide)
        size (int): Output size for both width and height (e.g., 224)

    Returns:
        PIL.Image: Cropped and resized image
    """
def crop_and_resize(image, size=CROP_SIZE, position="center"):
    """
    Crop a horizontal slice of the panorama (left, center, or right) and resize to square.

    Args:
        image (PIL.Image): Full panoramic image
        size (int): Output square size (e.g., 224)
        position (str): 'left', 'center', or 'right'

    Returns:
        PIL.Image: Processed 224×224 image
    """
    w, h = image.size

    # Focus on middle vertical band
    crop_height = int(h * 0.4)
    top = int(h * 0.3)
    bottom = top + crop_height

    # Select horizontal window
    crop_width = crop_height  # keep square crop
    if position == "left":
        left = 0
    elif position == "right":
        left = w - crop_width
    else:  # 'center'
        left = (w - crop_width) // 2
    right = left + crop_width

    cropped = image.crop((left, top, right, bottom))
    return cropped.resize((size, size), Image.BICUBIC)

def save_processed_image(image, city, season, sharpness, pano_id, lat, lon, date, save_dir):
    """
    Save processed image with full metadata in filename:
    data/{city}/{season}/{sharpness}/{panoid}_{lat}_{lon}_{date}_{season}_{sharpness}.jpg

    Args:
        image (PIL.Image): Processed image
        city (str): 'london_uk' or 'london_on'
        season (str): 'summer', 'fall', etc.
        sharpness (str): 'sharp' or 'blurry'
        pano_id (str): Panorama ID
        lat (float): Latitude of image
        lon (float): Longitude of image
        date (str): YYYY-MM
        save_dir (Path): Root path to data/ directory
    """
    target_dir = save_dir / city / season / sharpness
    target_dir.mkdir(parents=True, exist_ok=True)

    # Format metadata fields for filename
    lat_str = f"{lat:.5f}".replace('.', 'p')
    lon_str = f"{lon:.5f}".replace('.', 'p')
    date_str = date.replace("-", "")  # e.g., 202306
    filename = f"{pano_id}_{lat_str}_{lon_str}_{date_str}_{season}_{sharpness}.jpg"

    save_path = target_dir / filename

    try:
        image.save(save_path, "JPEG", quality=JPEG_QUALITY)
        print(f"Saved processed image to {save_path}")
    except Exception as e:
        print(f"Failed to save image: {e}")
