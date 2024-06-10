from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
from local_lib.common import common_height, common_width
from local_lib.images import generate_resized_dataset
    
class DataGeneratorTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, permutation, n_samples=5800, img_format=(common_width, common_height)):
        self.img_format = img_format
        self.n_samples = n_samples
        self.permutation = permutation

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        ar = generate_resized_dataset(self.permutation, self.img_format[0], self.img_format[1], self.n_samples)
        ar = ar[-self.n_samples:]
        return 