"""
Signal Inversion Module
=======================

This module extracts phase information from raw physiological signals and prepares
inputs for neural models. It corresponds to the first step in the framework.
"""

import numpy as np
from scipy import signal


class SignalInverter:
    """
    Extract phase information from raw signals and prepare inputs for neural models.
    
    This class processes raw data to extract phase information using
    Hilbert transform, estimate natural frequencies, and compute functional
    connectivity between channels.
    """
    
    def __init__(self, sampling_rate=256):
        """
        Initialize with sampling rate.
        
        Parameters:
        -----------
        sampling_rate : float
            Sampling rate of the data in Hz
        """
        self.sampling_rate = sampling_rate
        self.data = None
        self.phases = None
        self.frequencies = None
        self.connectivity_matrix = None
        
    def load_data(self, data):
        """
        Load data for processing.
        
        Parameters:
        -----------
        data : ndarray, shape (n_channels, n_samples)
            Signal data with channels as rows and time points as columns
        """
        self.data = data
        
    def preprocess(self, bandpass=(8, 13), notch=50):
        """
        Preprocess data with filters.
        
        Parameters:
        -----------
        bandpass : tuple, optional
            Bandpass filter range in Hz (default: alpha band 8-13 Hz)
        notch : int or None, optional
            Frequency for notch filter in Hz (default: 50 Hz for power line)
            
        Returns:
        --------
        filtered_data : ndarray, shape (n_channels, n_samples)
            Filtered data
        
        Raises:
        -------
        ValueError
            If no data is loaded
        """
        if self.data is None:
            raise ValueError("No data loaded. Use load_data() first.")
            
        n_channels, n_samples = self.data.shape
        filtered_data = np.zeros_like(self.data)
        
        # Notch filter for power line interference
        if notch:
            b_notch, a_notch = signal.iirnotch(notch, 30, self.sampling_rate)
            
        # Bandpass filter
        nyq = 0.5 * self.sampling_rate
        low, high = bandpass[0] / nyq, bandpass[1] / nyq
        b_bandpass, a_bandpass = signal.butter(3, [low, high], btype='band')
        
        for i in range(n_channels):
            # Apply notch filter if requested
            if notch:
                filtered_data[i, :] = signal.filtfilt(b_notch, a_notch, self.data[i, :])
            else:
                filtered_data[i, :] = self.data[i, :]
                
            # Apply bandpass filter
            filtered_data[i, :] = signal.filtfilt(b_bandpass, a_bandpass, filtered_data[i, :])
            
        return filtered_data
        
    def compute_hilbert_phase(self, filtered_data=None):
        """
        Extract phase using Hilbert transform with bandpass filtering.
        
        Parameters:
        -----------
        filtered_data : ndarray, shape (n_channels, n_samples), optional
            Pre-filtered data. If None, uses internally stored data
            
        Returns:
        --------
        phases : ndarray, shape (n_channels, n_samples)
            Instantaneous phase for each channel
            
        Raises:
        -------
        ValueError
            If no data is available for processing
        """
        if filtered_data is None:
            if self.data is None:
                raise ValueError("No data loaded. Use load_data() first.")
            filtered_data = self.preprocess()
        
        n_channels, n_samples = filtered_data.shape
        self.phases = np.zeros((n_channels, n_samples))
        
        for i in range(n_channels):
            # Apply Hilbert transform
            analytic_signal = signal.hilbert(filtered_data[i, :])
            
            # Extract instantaneous phase
            self.phases[i, :] = np.angle(analytic_signal)
            
        return self.phases
    
    def derive_natural_frequencies(self, phases=None):
        """
        Estimate natural frequency from phase time series.
        
        Parameters:
        -----------
        phases : ndarray, shape (n_channels, n_samples), optional
            Phase time series. If None, uses internally stored phases
            
        Returns:
        --------
        frequencies : ndarray, shape (n_channels,)
            Natural frequency for each channel in Hz
            
        Raises:
        -------
        ValueError
            If no phase data is available
        """
        if phases is None:
            if self.phases is None:
                raise ValueError("No phase data available. Extract phases first.")
            phases = self.phases
        
        n_channels, n_samples = phases.shape
        self.frequencies = np.zeros(n_channels)
        
        for i in range(n_channels):
            # Unwrap phase to prevent discontinuities
            unwrapped_phase = np.unwrap(phases[i, :])
            
            # Calculate phase differences
            diff_phase = np.diff(unwrapped_phase)
            
            # Convert to frequency (Hz)
            inst_freq = diff_phase * self.sampling_rate / (2 * np.pi)
            
            # Get median frequency (more robust than mean)
            self.frequencies[i] = np.median(inst_freq)
                
        return self.frequencies
    
    def assess_connectivity(self, phases=None, method='plv', threshold=0.3):
        """
        Estimate functional connectivity from phases.
        
        Parameters:
        -----------
        phases : ndarray, shape (n_channels, n_samples), optional
            Phase time series. If None, uses internally stored phases
        method : str, optional
            Method for connectivity estimation ('plv', 'pli', 'wpli')
        threshold : float, optional
            Threshold for binary connectivity matrix
            
        Returns:
        --------
        connectivity_matrix : ndarray, shape (n_channels, n_channels)
            Estimated connectivity matrix
            
        Raises:
        -------
        ValueError
            If no phase data is available or unknown connectivity method
        """
        if phases is None:
            if self.phases is None:
                raise ValueError("No phase data available. Extract phases first.")
            phases = self.phases
            
        n_channels, n_samples = phases.shape
        self.connectivity_matrix = np.zeros((n_channels, n_channels))
        
        if method == 'plv':
            # Phase Locking Value
            for i in range(n_channels):
                for j in range(n_channels):
                    if i != j:
                        phase_diff = phases[i, :] - phases[j, :]
                        self.connectivity_matrix[i, j] = np.abs(np.mean(np.exp(1j * phase_diff)))
        
        elif method == 'pli':
            # Phase Lag Index
            for i in range(n_channels):
                for j in range(n_channels):
                    if i != j:
                        phase_diff = phases[i, :] - phases[j, :]
                        self.connectivity_matrix[i, j] = np.abs(np.mean(np.sign(np.sin(phase_diff))))
        
        elif method == 'wpli':
            # Weighted Phase Lag Index
            for i in range(n_channels):
                for j in range(n_channels):
                    if i != j:
                        phase_diff = phases[i, :] - phases[j, :]
                        imag_csd = np.sin(phase_diff)
                        self.connectivity_matrix[i, j] = np.abs(np.mean(imag_csd)) / np.mean(np.abs(imag_csd))
        
        else:
            raise ValueError(f"Unknown connectivity method: {method}")
            
        # Apply threshold if needed
        if threshold:
            self.connectivity_matrix = (self.connectivity_matrix > threshold).astype(float)
            
        return self.connectivity_matrix