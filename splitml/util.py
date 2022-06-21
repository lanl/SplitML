"""
Utility functions.

Author: Natalie Klein
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_complex_ts(t, y, ax=None, **kwargs):
    """
    Plot complex-valued time series. 
    """
    if ax is None:
        ax = plt.gca()
    for f, c in zip([np.real, np.imag], ['k', 'r']):
        ax.plot(t, f(y), c, **kwargs, label=f.__name__)

def compute_snr(sig, noise):
    """
    Compute SNR empirically given signal and noise data.
    """
    noise_var = np.var(noise - np.mean(noise, 1, keepdims=True), 1)
    sig_var = np.var(sig - np.mean(sig, 1, keepdims=True), 1)
    return 10.0 * np.log10(sig_var/noise_var)