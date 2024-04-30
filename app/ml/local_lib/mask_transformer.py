from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
from local_lib.common import common_height, common_width
    
class MaskSeuilTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, strategy='baseline'):
        self.strategy = strategy

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        transformed_images = []
        mask = []

        if self.strategy == 'baseline':
            mask = pd.read_csv("../../datasets/chest_Xray/_processed_resize_small/data_mask_seuil.csv", delimiter=",") 
        elif self.strategy == 'geometry':
            mask = pd.read_csv("../../datasets/chest_Xray/_processed_resize_small/data_mask_triangle.csv", delimiter=",") 
        elif self.strategy == 'combined':
            mask = pd.read_csv("../../datasets/chest_Xray/_processed_resize_small/data_mask_seuil_triangle.csv", delimiter=",") 
        else:
            raise ValueError("Unsupported strategy")

        for array_image in X:
            reshaped_data = array_image.reshape(common_height, common_width)
            masked_image_array = np.where(mask == 1, reshaped_data, 0)
            transformed_images.append(masked_image_array.flatten())
            
        return np.array(transformed_images)