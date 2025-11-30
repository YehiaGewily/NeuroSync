"""
Signal Generator Module
===================

This module generates signal time series from oscillator network phases.
It corresponds to the sixth step in the framework.
"""

import numpy as np
from scipy.linalg import svd
from scipy import signal


class SignalGenerator:
    """
    Generate signal time series from oscillator network phases.
    
    This class converts oscillatory phases from the oscillator network into signal-like
    time series, with options for applying spatial filtering and computing spectral features.
    """
    
    def __init__(self, oscillator_network=None):
        """
        Initialize generator with oscillator network.
        
        Parameters:
        -----------
        oscillator_network : OscillatorNetwork or DynamicOscillatorNetwork, optional
            Model providing phases
        """
        self.model = oscillator_network
        self.signal_data = None
        self.svd_components = None
        self.spatial_maps = None
        
    def set_oscillator_network(self, model):
        """
        Set the oscillator network model.
        
        Parameters:
        -----------
        model : OscillatorNetwork or DynamicOscillatorNetwork
            Model to use for generating signals
        """
        self.model = model
        
    def phase_to_signal(self, phases=None, amplitudes=None):
        """
        Convert phases to signal-like time series.
        
        Parameters:
        -----------
        phases : ndarray, shape (n_time_points, n_nodes), optional
            Phases from oscillator network
        amplitudes : ndarray, shape (n_time_points, n_nodes), optional
            Amplitudes (default: all 1.0)
            
        Returns:
        --------
        signal_data : ndarray, shape (n_nodes, n_time_points)
            Generated signal-like time series
            
        Raises:
        -------
        ValueError
            If no phases are available
        """
        if phases is None:
            if self.model is None or self.model.phases is None:
                raise ValueError("No phases available. Provide phases or set oscillator network.")
            phases = self.model.phases
            
        n_time_points, n_nodes = phases.shape
        
        # Use provided amplitudes or set to 1.0
        if amplitudes is None:
            amplitudes = np.ones_like(phases)
            
        # Generate signals
        self.signal_data = np.zeros((n_nodes, n_time_points))
        
        for i in range(n_nodes):
            self.signal_data[i, :] = amplitudes[:, i] * np.cos(phases[:, i])
            
        return self.signal_data
    
    def apply_svd(self, signal_data=None, n_components=None):
        """
        Apply Singular Value Decomposition to signal data.
        
        Parameters:
        -----------
        signal_data : ndarray, shape (n_channels, n_time_points), optional
            Signal data to decompose
        n_components : int, optional
            Number of components to retain (default: keep 99% variance)
            
        Returns:
        --------
        reconstructed_data : ndarray, shape (n_channels, n_time_points)
            Reconstructed signal data using selected components
            
        Raises:
        -------
        ValueError
            If no signal data is available
        """
        if signal_data is None:
            if self.signal_data is None:
                raise ValueError("No signal data available. Generate or provide data first.")
            signal_data = self.signal_data
            
        # Apply SVD
        U, S, Vt = svd(signal_data, full_matrices=False)
        
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
    
    def compute_psd(self, signal_data=None, fs=256, fmin=1, fmax=45, n_fft=512):
        """
        Compute Power Spectral Density.
        
        Parameters:
        -----------
        signal_data : ndarray, shape (n_channels, n_time_points), optional
            Signal data
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
            If no signal data is available
        """
        if signal_data is None:
            if self.signal_data is None:
                raise ValueError("No signal data available. Generate or provide data first.")
            signal_data = self.signal_data
            
        n_channels, n_times = signal_data.shape
        
        # Initialize output
        freqs = np.linspace(0, fs/2, n_fft//2 + 1)
        mask = (freqs >= fmin) & (freqs <= fmax)
        freqs = freqs[mask]
        psd = np.zeros((n_channels, np.sum(mask)))
        
        # Compute PSD for each channel
        for ch in range(n_channels):
            f, Pxx = signal.welch(signal_data[ch, :], fs=fs, nperseg=n_fft, noverlap=n_fft//2)
            psd[ch, :] = Pxx[mask]
            
        return freqs, psd