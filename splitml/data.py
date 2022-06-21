"""
Noisy data set generation.

Author: Natalie Klein

TODO: Add other noise generation models.
"""

import numpy as np

from splitml.signal import VoigtSignal

def sig_gen(t, df):
    """
    Generate collection of Voigt time series signals based on Pandas data frame of parameter values.
    Args:
        t: vector of time points
        df: Pandas data frame containing signal parameter columns A, w, T2, phi, sigma, C, with N rows
    Returns:
        np.ndarray of signals, shape (N, len(t))
    """
    sigs = np.zeros((df.shape[0], len(t)), dtype=complex)
    for i in range(df.shape[0]):
        sigs[i, :] = VoigtSignal(df.w.iloc[i], df.T2.iloc[i], df.A.iloc[i], df.phi.iloc[i], df.sigma.iloc[i], df.C.iloc[i]).time_signal(t)
    return sigs
    
def wn_gen(t, N, sigma=1.0):
    """
    Generate complex Gaussian white noise time series of shape (N, t) with total variance sigma.
    """
    return np.random.normal(0, scale=sigma/np.sqrt(2), size=(N, len(t))) + 1j* np.random.normal(0, scale=sigma/np.sqrt(2), size=(N, len(t)))

def split_noise(noise, n_timesteps):
    """
    Split noise data array along time axis to increase number of examples (with shorter duration).
    """
    nt = noise.shape[1]
    nseg = nt // n_timesteps
    noise = noise[:, :(nseg*n_timesteps)]
    snoise = np.stack(np.hsplit(noise, nseg), axis=1)
    rsnoise = np.reshape(snoise, (noise.shape[0]*nseg, -1))
    return rsnoise