"""
Generate complex NQR signals using Voigt model.

Author: Natalie Klein
"""

import numpy as np
from scipy.special import wofz

class VoigtSignal:
    """
    Voigt signal model.
    """

    def __init__(self, w, T2, A, phi, sigma, C):
        """
        Voigt signal model. If sigma=np.inf, recover FID signal model.
        Args: 
            w: frequency (Hz)
            T2: time decay constant
            A: amplitude constant
            phi: phase shift constant
            sigma: shape constant
            C: constant shift
        """
        super(VoigtSignal, self).__init__()
        self.w = w
        self.T2 = T2
        self.A = A
        self.phi = phi
        self.sigma = sigma
        self.C = C

    def time_signal(self, t):
        """
        Return complex-valued time-domain signal given time vector t.
        """
        return self.A * np.exp(-0.5*(t/self.sigma)**2) * np.exp(-t/self.T2) * np.exp(1j*(-2*np.pi*self.w*t + self.phi)) + self.C

    def freq_signal(self, f_vec): 
        """
        Return real-valued frequency-domain representation given frequency vector f_vec.
        """
        fit = np.real(wofz((f_vec-self.w + 1j*self.T2)/self.sigma/np.sqrt(2)))/self.sigma
        return self.A*fit/np.max(fit)

