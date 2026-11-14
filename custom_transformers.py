
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class InvalidWatchTimeHandler(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        X["avg_watch_time_per_day"] = (
            X["avg_watch_time_per_day"]
            .mask(X["avg_watch_time_per_day"] > 24, np.nan)
        )

        return X
