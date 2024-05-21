import os
import pandas as pd
import random
from PIL import Image
from .images import image_to_dataframe
import numpy as np

def retreiveDatasetFromCsv(n_samples=5856, src="../../datasets/chest_Xray/_processed_resize_small/_processed_dfs"):
    dfs_folder = src
    dataset = pd.DataFrame(columns=['pixel_value', 'class']) 
    dfs_files = [f for f in os.listdir(dfs_folder) if os.path.isfile(os.path.join(dfs_folder, f))]
    # random.shuffle(dfs_files)

    for df_file in dfs_files[:n_samples]:
        df = pd.read_csv(os.path.join(dfs_folder, df_file))
        found_class = 0
        if "bacteria" in df_file:
            found_class = 2
        elif "virus" in df_file:
            found_class = 1
        else:
            found_class = 0

        pixel_values = df.values.flatten()
        new_row = {'pixel_value': [pixel_values], 'class': found_class}
        dataset = pd.concat([dataset, pd.DataFrame(new_row)], ignore_index=True)
    return dataset

def randomRetreiveDatasetFromCsv(permutation, n_samples=5856):
    dfs_folder = "../../datasets/chest_Xray/_processed_resize_small/_processed_dfs"
    dataset = pd.DataFrame(columns=['pixel_value', 'class']) 
    dfs_files = [f for f in os.listdir(dfs_folder) if os.path.isfile(os.path.join(dfs_folder, f))]

    dfs_files = np.array(dfs_files)[permutation].tolist()

    for df_file in dfs_files[:n_samples]:
        df = pd.read_csv(os.path.join(dfs_folder, df_file))
        found_class = 0
        if "bacteria" in df_file:
            found_class = 2
        elif "virus" in df_file:
            found_class = 1
        else:
            found_class = 0

        pixel_values = df.values.flatten()
        new_row = {'pixel_value': [pixel_values], 'class': found_class}
        dataset = pd.concat([dataset, pd.DataFrame(new_row)], ignore_index=True)
    return dataset