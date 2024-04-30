from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer

class TesterTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, strategy='StandardScaler'):
        self.strategy = strategy

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.strategy == 'StandardScaler':
            return StandardScaler().fit_transform(X)
        elif self.strategy == 'MinMaxScaler':
           return MinMaxScaler().fit_transform(X)
        elif self.strategy == 'Normalizer':
            return Normalizer().fit_transform(X)
        else:
            return StandardScaler().fit_transform(X)

        