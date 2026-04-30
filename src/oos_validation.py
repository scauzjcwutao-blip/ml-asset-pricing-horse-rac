"""
Out-of-Sample Validation Module
Implements rolling-window cross-validation for time series data.
"""

import pandas as pd
import numpy as np


class OOSValidator:
    """
    Rolling-window out-of-sample validator for time series.
    Implements strict forward-looking validation to prevent look-ahead bias.
    """

    def __init__(self, n_splits=5, test_size=252, min_train_size=None):
        """
        Initialize the OOS validator.

        Parameters:
        -----------
        n_splits : int
            Number of train/test splits
        test_size : int
            Number of observations in each test set (e.g., 252 for ~1 year of trading days)
        min_train_size : int, optional
            Minimum number of observations in training set. If None, uses expanding window.
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.min_train_size = min_train_size

    def split(self, X, y):
        """
        Generate rolling-window train/test splits.

        Parameters:
        -----------
        X : pd.DataFrame or np.array
            Feature matrix
        y : pd.Series or np.array
            Target variable

        Yields:
        -------
        X_train, X_test, y_train, y_test : train/test splits
        """
        n_samples = len(X)

        # Calculate total samples needed
        total_needed = self.min_train_size or 0 + self.n_splits * self.test_size
        if total_needed > n_samples:
            raise ValueError(f"Not enough samples ({n_samples}) for {self.n_splits} splits "
                           f"with test_size={self.test_size}")

        # Calculate starting point for first split
        if self.min_train_size is not None:
            start_idx = self.min_train_size
        else:
            # Expanding window: first split uses roughly n_splits * test_size for training
            start_idx = n_samples - self.n_splits * self.test_size

        # Generate splits
        for i in range(self.n_splits):
            test_start = start_idx + i * self.test_size
            test_end = min(test_start + self.test_size, n_samples)

            if test_start >= n_samples:
                break

            if isinstance(X, pd.DataFrame):
                X_train = X.iloc[:test_start]
                X_test = X.iloc[test_start:test_end]
            else:
                X_train = X[:test_start]
                X_test = X[test_start:test_end]

            if isinstance(y, pd.Series):
                y_train = y.iloc[:test_start]
                y_test = y.iloc[test_start:test_end]
            else:
                y_train = y[:test_start]
                y_test = y[test_start:test_end]

            yield X_train, X_test, y_train, y_test

    def get_n_splits(self, X=None, y=None, groups=None):
        """Return number of splits."""
        return self.n_splits
