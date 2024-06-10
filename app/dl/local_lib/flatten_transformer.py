from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
    
class FlattenTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        transformed_images = []
        for array_image in X:
            transformed_images.append(array_image.flatten())
        return np.array(transformed_images)