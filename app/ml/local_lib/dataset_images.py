import os
import pandas as pd
import random
from PIL import Image
from .images import image_to_dataframe

def retreiveDatasetFromImages(n_samples=700):
    imgs_folder = "../../datasets/chest_Xray/_processed_resize/_processed_imgs"
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
            # Convert the resized image to grayscale DataFrame
            img_df = image_to_dataframe(img)
            pixel_values = img_df.values.flatten()
            new_row = {'pixel_value': [pixel_values], 'class': found_class}
            dataset = pd.concat([dataset, pd.DataFrame(new_row)], ignore_index=True)
    return dataset