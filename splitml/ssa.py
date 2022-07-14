"""
Singular spectrum analysis for complex-valued signals.

Author: Natalie Klein
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import svd

def diag_avg(X, n_timestamps, window_size, n_windows, gap):
    X_new = np.empty(n_timestamps, dtype=complex)
    first_row = [(0, col) for col in range(n_windows)]
    last_col = [(row, n_windows - 1) for row in range(1, window_size)]
    indices = first_row + last_col
    for (j, k) in indices:
        X_new[j + k] = np.diag(X[:, ::-1], gap - j - k - 1).mean()
    return X_new

def lag_time_series(y_orig, n_lags):
    y = y_orig[:, n_lags:]
    x = np.zeros((y.shape[0], y.shape[1], n_lags), dtype=complex)
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            x[i, j, :] = y_orig[i, j:(j+n_lags)]
    return x, y

class SSA():

    def __init__(self, n_lags):
        """
        Initialize SSA.
            n_lags (int): window size for time series
        """
        super(SSA, self).__init__()
        self.n_lags = n_lags
        self.U = None
        self.S = None
        self.Vh = None

    def fit(self, X):
        """
        Fit SSA.
            X (ndarray): array of time series, shape (n, len(time))
        """
        self.X_len = X.shape[1]
        X_lag, _ = lag_time_series(X, self.n_lags).squeeze()
        U, S, Vh = svd(X_lag, full_matrices=False)
        self.U = U
        self.S = S
        self.Vh = Vh

    def get_basis(self):
        return self.Vh

    def transform(self, X, components=None):
        """
        Transform time series to SSA latent/projection space.
            X (ndarray): array of time series, shape (n, len(time))
            components (list or None): list of component indices to use or None to use all
        """
        if components is None:
            X_proj = X @ Vh.conj().T
        else: 
            X_proj = X @ Vh[components, :].conj().T
        return X_proj

    def predict(self, X, components=None):
        """
        Predict/reconstruct time series using SSA components. 
            X (ndarray): array of time series, shape (n, len(time))
            components (list or None): list of component indices to use or None to use all
        """
        X_proj = self.transform(X, components)
        if components is None:
            X_pred = X_proj @ Vh
        else:
            X_pred = X_proj @ Vh[components, :]
        n_win = X_pred.shape[0]
        X_pred = diag_avg(X_pred.T, self.X_len, self.n_lags, n_win, n_win)
        return X_pred


    
