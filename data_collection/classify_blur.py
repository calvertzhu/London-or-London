import numpy as np
import cv2
from PIL import Image
from data_collection.config import BLUR_THRESHOLD

def compute_laplacian_variance(image: Image.Image) -> float:
    """
    Compute Laplacian variance of an image to measure sharpness.

    Args:
        image (PIL.Image): Input image, assumed RGB or grayscale

    Returns:
        float: Variance of Laplacian (higher = sharper)
    """
    gray = np.array(image.convert("L"))  # Convert to grayscale
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return lap_var

def classify_sharpness(image: Image.Image, threshold: float = BLUR_THRESHOLD) -> str:
    """
    Classify an image as 'sharp' or 'blurry' based on Laplacian variance.

    Args:
        image (PIL.Image): Processed (224×224) image
        threshold (float): Laplacian variance threshold

    Returns:
        str: 'sharp' or 'blurry'
    """
    var = compute_laplacian_variance(image)
    return "sharp" if var >= threshold else "blurry"