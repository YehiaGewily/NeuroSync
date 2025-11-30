"""
EEG Generator Module
===================

This module generates EEG time series from Kuramoto model phases.
It corresponds to the sixth step in the framework.
"""

import numpy as np
from scipy.linalg import svd
from scipy import signal


class EEGGenerator:
    """
    Generate EEG time series from Kuramoto model phases.
    
    This class converts oscillatory phases from the Kuramoto model into EEG-like
    signals, with options for applying spatial filtering and computing spectral features.
    """
    
    def __init__(self, kuramoto_model=None):
        """
        Initialize generator with Kuramoto model.
        
        Parameters:
        -----------
        kuramoto_model : KuramotoModel or TimeVaryingKuramoto, optional
            Model providing phases
        """
        self.model = kuramoto_model
        self.eeg_data = None
        self.svd_components = None
        self.spatial_maps = None
        
    def set_kuramoto_model(self, model):
        """
        Set the Kuramoto model.
        
        Parameters:
        -----------
        model : KuramotoModel or TimeVaryingKuramoto
            Model to use for generating EEG
        """
        self.model = model
        
    def phase_to_signal(self, phases=None, amplitudes=None):
        """
        Convert phases to EEG-like signals.
        
        Parameters:
        -----------
        phases : ndarray, shape (n_time_points, n_oscillators), optional
            Phases from Kuramoto model
        amplitudes : ndarray, shape (n_time_points, n_oscillators), optional
            Amplitudes (default: all 1.0)
            
        Returns:
        --------
        eeg_data : ndarray, shape (n_oscillators, n_time_points)
            Generated EEG-like signals
            
        Raises:
        -------
        ValueError
            If no phases are available
        """
        if phases is None:
            if self.model is None or self.model.phases is None:
                raise ValueError("No phases available. Provide phases or set Kuramoto model.")
            phases = self.model.phases
            
        n_time_points, n_oscillators = phases.shape
        
        # Use provided amplitudes or set to 1.0
        if amplitudes is None:
            amplitudes = np.ones_like(phases)
            
        # Generate EEG signals
        self.eeg_data = np.zeros((n_oscillators, n_time_points))
        
        for i in range(n_oscillators):
            self.eeg_data[i, :] = amplitudes[:, i] * np.cos(phases[:, i])
            
        return self.eeg_data
    
    def apply_svd(self, eeg_data=None, n_components=None):
        """
        Apply Singular Value Decomposition to EEG data.
        
        Parameters:
        -----------
        eeg_data : ndarray, shape (n_channels, n_time_points), optional
            EEG data to decompose
        n_components : int, optional
            Number of components to retain (default: keep 99% variance)
            
        Returns:
        --------
        reconstructed_data : ndarray, shape (n_channels, n_time_points)
            Reconstructed EEG data using selected components
            
        Raises:
        -------
        ValueError
            If no EEG data is available
        """
        if eeg_data is None:
            if self.eeg_data is None:
                raise ValueError("No EEG data available. Generate or provide data first.")
            eeg_data = self.eeg_data
            
        # Apply SVD
        U, S, Vt = svd(eeg_data, full_matrices=False)
        
        # Determine number of components to keep
        if n_components is None:
            # Keep components explaining 99% of variance
            var_explained = np.cumsum(S**2) / np.sum(S**2)
            n_components = np.where(var_explained >= 0.99)[0][0] + 1
            
        # Reconstruct with selected components
        self.svd_components = Vt[:n_components, :]
        self.spatial_maps = U[:, :n_components]
        S_reduced = np.diag(S[:n_components])
        
        reconstructed_data = np.dot(self.spatial_maps, np.dot(S_reduced, self.svd_components))
        
        return reconstructed_data
    
    def compute_psd(self, eeg_data=None, fs=256, fmin=1, fmax=45, n_fft=512):
        """
        Compute Power Spectral Density.
        
        Parameters:
        -----------
        eeg_data : ndarray, shape (n_channels, n_time_points), optional
            EEG data
        fs : float, optional
            Sampling frequency
        fmin, fmax : float, optional
            Minimum and maximum frequencies to consider
        n_fft : int, optional
            Number of FFT points
            
        Returns:
        --------
        freqs : ndarray
            Frequency bins
        psd : ndarray, shape (n_channels, n_freqs)
            Power spectral density for each channel
            
        Raises:
        -------
        ValueError
            If no EEG data is available
        """
        if eeg_data is None:
            if self.eeg_data is None:
                raise ValueError("No EEG data available. Generate or provide data first.")
            eeg_data = self.eeg_data
            
        n_channels, n_times = eeg_data.shape
        
        # Initialize output
        freqs = np.linspace(0, fs/2, n_fft//2 + 1)
        mask = (freqs >= fmin) & (freqs <= fmax)
        freqs = freqs[mask]
        psd = np.zeros((n_channels, np.sum(mask)))
        
        # Compute PSD for each channel
        for ch in range(n_channels):
            f, Pxx = signal.welch(eeg_data[ch, :], fs=fs, nperseg=n_fft, noverlap=n_fft//2)
            psd[ch, :] = Pxx[mask]
            
        return freqs, psd