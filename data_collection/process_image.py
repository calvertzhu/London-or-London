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
    w, h = image.size
    min_dim = min(w, h)
    
    # Center crop (square from center)
    left = (w - min_dim) // 2
    top = (h - min_dim) // 2
    right = left + min_dim
    bottom = top + min_dim

    image_cropped = image.crop((left, top, right, bottom))
    image_resized = image_cropped.resize((size, size), Image.BICUBIC)
    return image_resized

def save_processed_image(image, city, season, sharpness, pano_id, save_dir):
    """
    Save the processed image to data/{city}/{season}/{sharpness}/{pano_id}.jpg

    Args:
        image (PIL.Image): Processed (cropped + resized) image
        city (str): 'london_uk' or 'london_on'
        season (str): 'summer', 'fall', etc.
        sharpness (str): 'sharp' or 'blurry'
        pano_id (str): ID used as filename
        save_dir (Path): Root path to data/ directory
    """
    target_dir = save_dir / city / season / sharpness
    target_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{pano_id}.jpg"
    save_path = target_dir / filename

    try:
        image.save(save_path, "JPEG", quality=JPEG_QUALITY)
        print(f"Saved processed image to {save_path}")
    except Exception as e:
        print(f"Failed to save image: {e}")