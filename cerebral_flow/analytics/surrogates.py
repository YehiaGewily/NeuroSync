"""
Surrogate Data Generation Module
===============================

This module provides tools for generating surrogate data to test the statistical
significance of results.
"""

import numpy as np
from scipy import signal

class SurrogateGenerator:
    """
    Generate surrogate data for statistical testing.
    
    Methods:
    --------
    phase_shuffle : Preserves power spectrum but destroys phase relationships.
    time_shift : Shifts time series to destroy instantaneous coupling.
    """
    
    def __init__(self, data, sampling_rate=256):
        """
        Initialize generator.
        
        Parameters:
        -----------
        data : ndarray, shape (n_channels, n_samples)
            Original signal data
        sampling_rate : float
            Sampling rate in Hz
        """
        self.data = data
        self.fs = sampling_rate
        self.n_channels, self.n_samples = data.shape
        
    def phase_shuffle(self, n_surrogates=1):
        """
        Generate phase-shuffled surrogates.
        
        This method computes the FFT of the data, randomizes the phases
        (while preserving symmetry for real signals), and computes the inverse FFT.
        
        Parameters:
        -----------
        n_surrogates : int
            Number of surrogate datasets to generate
            
        Yields:
        -------
        surrogate : ndarray
            A single surrogate dataset
        """
        for _ in range(n_surrogates):
            surrogate = np.zeros_like(self.data)
            
            for ch in range(self.n_channels):
                # FFT
                x_fft = np.fft.rfft(self.data[ch])
                
                # Randomize phases
                random_phases = np.random.uniform(0, 2*np.pi, len(x_fft))
                random_phases[0] = 0  # DC component
                if len(self.data[ch]) % 2 == 0:
                    random_phases[-1] = 0 # Nyquist (if even)
                    
                # Apply new phases
                x_fft_shuffled = np.abs(x_fft) * np.exp(1j * random_phases)
                
                # Inverse FFT
                surrogate[ch] = np.fft.irfft(x_fft_shuffled, n=self.n_samples)
                
            yield surrogate

    def time_shift(self, n_surrogates=1, min_shift=0.1):
        """
        Generate time-shifted surrogates.
        
        Circularly shifts each channel by a random amount.
        
        Parameters:
        -----------
        n_surrogates : int
            Number of surrogates
        min_shift : float
            Minimum shift as a fraction of total duration
            
        Yields:
        -------
        surrogate : ndarray
        """
        min_samples = int(min_shift * self.n_samples)
        
        for _ in range(n_surrogates):
            surrogate = np.zeros_like(self.data)
            
            for ch in range(self.n_channels):
                shift = np.random.randint(min_samples, self.n_samples - min_samples)
                surrogate[ch] = np.roll(self.data[ch], shift)
                
            yield surrogate

class SignificanceTester:
    """
    Test statistical significance against surrogates.
    """
    
    @staticmethod
    def calculate_p_value(observed_metric, surrogate_metrics, tail='both'):
        """
        Calculate p-value of observed metric given distribution of surrogate metrics.
        
        Parameters:
        -----------
        observed_metric : float
            Value measured on real data
        surrogate_metrics : list or ndarray
            Values measured on surrogate data
        tail : str, 'left', 'right', or 'both'
            Tail of the test
            
        Returns:
        --------
        p_value : float
        z_score : float
        """
        surrogate_metrics = np.array(surrogate_metrics)
        mean_surr = np.mean(surrogate_metrics)
        std_surr = np.std(surrogate_metrics)
        
        # Z-score
        z_score = (observed_metric - mean_surr) / (std_surr + 1e-10)
        
        # P-value
        if tail == 'right':
            p_value = np.mean(surrogate_metrics >= observed_metric)
        elif tail == 'left':
            p_value = np.mean(surrogate_metrics <= observed_metric)
        else:
            # Two-tailed
            p_value = np.mean(np.abs(surrogate_metrics - mean_surr) >= np.abs(observed_metric - mean_surr))
            
        return p_value, z_score
