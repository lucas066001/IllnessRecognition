import os
import pandas as pd
from PIL import Image

def resize_image(image_path, size):
    """
    Resize the image to the specified size while keeping the aspect ratio.
    """
    img = Image.open(image_path)
    img_resized = img.resize(size, Image.LANCZOS)
    return img_resized

def image_to_dataframe(image):
    """
    Convert an image to a DataFrame representing its grayscale values.
    """
    img_gray = image.convert("L")
    data = list(img_gray.getdata())
    df = pd.DataFrame(data, columns=["Pixel_Value"])
    return df