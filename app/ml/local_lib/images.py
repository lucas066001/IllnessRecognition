import os
import pandas as pd
from PIL import Image
import random
import numpy as np

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


def image_to_array(image):
    """
    Convert an image to a DataFrame representing its grayscale values.
    """
    img_gray = image.convert("L")
    data = list(img_gray.getdata())
    array = np.array(data)
    return array

def generate_resized_dataset(max_width, max_height, n_samples=5800):
    """
    Generate a dataset with given image size and samples
    """
    imgs_folder = "../../datasets/chest_Xray/raw_data"

    dataset = pd.DataFrame(columns=['pixel_value', 'class']) 

    imgs_files = [f for f in os.listdir(imgs_folder) if os.path.isfile(os.path.join(imgs_folder, f))]
    
    random.shuffle(imgs_files)

    for img_file in imgs_files[:n_samples]:
        found_class = 0
        if "bacteria" in img_file:
            found_class = 2
        elif "virus" in img_file:
            found_class = 1
        else:
            found_class = 0

        with Image.open(os.path.join(imgs_folder, img_file)) as img:
            #Resize image
            img_resized = img.resize((max_width, max_height), Image.LANCZOS)
            # Convert the resized image to grayscale DataFrame
            df = image_to_dataframe(img_resized)
            new_row = {'pixel_value': df.values, 'class': found_class}
            dataset = pd.concat([dataset, pd.DataFrame(new_row)], ignore_index=True)
    
    return dataset