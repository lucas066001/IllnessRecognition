from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
    
class NonZeroTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        transformed_images = []
        for array_image in X:
            # Convert the array to a floating-point type
            array = array_image.astype(np.float64)

            # Replace zeros with 0.01
            array[array == 0] = 1e-15

            transformed_images.append(array)
        return np.array(transformed_images)