from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
    
class DimensionTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        transformed_images = []
        for array_image in X:
            image = array_image[array_image != 0]
            # print(image.shape)
            transformed_images.append(image)
        
        result = np.array(transformed_images)
        return result