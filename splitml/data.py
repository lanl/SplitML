"""
Noisy data set generation.

Authors: Natalie Klein and Amber Day

"""

import numpy as np
import pandas as pd

from splitml.util import compute_snr
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

def synthetic_data_gen(N = 10000, nt = 1000, fs = 1./1.8e-05, w_range = [-50,50], 
                        phi_range = [-np.pi, np.pi], T2_range = [1e-3, 1e-2],
                        sigma_range = [1e-3, 1e-2], A_range = [0.1,1], s = 1, seed=42):
    """
    Generating synthetic data with white noise
    
    Inputs
        N : number of examples
        nt : number of time points
        fs : sampling rate
        w_range : upper and lower bounds for generating the frequency, w
        phi_range : upper and lower bounds for generating the phase offset, phi
        T2_range & sigma_range : upper and lower bounds for generating the rate of decay, T2 & sigma
        A_range : upper and lower bounds for generating the initial amplitude, A
        s : variance for white noise generation
        seed: random seed
        data_split : data is split into three sets (60% training, 20% testing, 20% validation)
    Outputs
        clean_training : synthetic dataset of signal without noise (0.6*N samples)
        noisy_training : synthetic dataset of signal with noise (0.6*N samples)
        clean_validation : synthetic dataset of signal without noise (0.2*N samples)
        noisy_validation : synthetic dataset of signal with noise (0.2*N samples)
        clean_testing : synthetic dataset of signal without noise (0.2*N samples)
        noisy_testing : synthetic dataset of signal with noise (0.2*N samples)
        sig_params_training :  parameters used for generating training dataset
        sig_params_validation : parameters used for generating validation dataset
        sig_params_testing : parameters used for generating testing dataset
        t : corresponding time vector for samples

    """

    # Initializing Parameters
    np.random.seed(seed)
    w = np.random.uniform(w_range[0], w_range[1], N)
    phi = np.random.uniform(phi_range[0], phi_range[1], N)
    T2 = np.random.uniform(T2_range[0], T2_range[1], N)
    sigma = np.random.uniform(sigma_range[0], sigma_range[1], N)
    A = np.random.uniform(A_range[0], A_range[1], N)
    C = np.zeros(N)
    df = pd.DataFrame({'w':w, 'phi':phi, 'T2':T2, 'A':A, 'sigma':sigma, 'C':C})

    # Signal Generation
    t = np.linspace(0, nt/fs, nt)
    sigs = sig_gen(t, df)

    # Generate White Noise
    noise = wn_gen(t, N, sigma = s)
    snrs = compute_snr(sigs, noise)
    df['snr'] = snrs

    # Save Clean/Noisy and Training/Validation separately 
    data_split1 = round(0.6*N) 
    data_split2 = round(0.8*N)
    
    clean_training = sigs[:data_split1,]
    noisy_training = noise[:data_split1,]
    sig_params_training = df.iloc[:data_split1,] # Parameters for generating signal
    
    clean_validation = sigs[data_split1:data_split2,]
    noisy_validation = noise[data_split1:data_split2,]
    sig_params_validation = df.iloc[data_split1:data_split2,]
    
    clean_testing = sigs[data_split2:,]
    noisy_testing = noise[data_split2:,]
    sig_params_testing = df.iloc[data_split2:,]
    
    return (clean_training, noisy_training, clean_validation, noisy_validation, 
            clean_testing, noisy_testing, sig_params_training, 
            sig_params_validation, sig_params_testing, t)