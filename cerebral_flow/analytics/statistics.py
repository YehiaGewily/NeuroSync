"""
Statistical Analysis Module
=========================

This module performs statistical similarity analysis to update oscillator network parameters.
It corresponds to the eighth step in the framework.
"""

import numpy as np
from scipy import signal
from typing import Tuple, Dict, Optional, Union, List


class StatisticalAnalyzer:
    """
    Perform statistical similarity analysis to update oscillator network.
    
    This class provides methods to analyze the statistical properties of real and
    generated signals, and to derive parameter updates for the oscillator network
    based on the differences observed.
    """
    
    def __init__(self, real_signal: Optional[np.ndarray] = None, 
                 generated_signal: Optional[np.ndarray] = None):
        """
        Initialize analyzer with real and generated signals.
        
        Parameters
        ----------
        real_signal : ndarray, shape (n_channels, n_time_points), optional
            Real signal data
        generated_signal : ndarray, shape (n_channels, n_time_points), optional
            Generated signal data
        """
        self.real_signal = real_signal
        self.generated_signal = generated_signal
        
    def set_data(self, real_signal: np.ndarray, generated_signal: np.ndarray) -> None:
        """
        Set the real and generated signal data.
        
        Parameters
        ----------
        real_signal : ndarray, shape (n_channels, n_time_points)
            Real signal data
        generated_signal : ndarray, shape (n_channels, n_time_points)
            Generated signal data
        """
        self.real_signal = real_signal
        self.generated_signal = generated_signal
        self._validate_data()
        
    def _validate_data(self) -> None:
        """
        Validate that data is properly set and dimensions match.
        
        Raises
        ------
        ValueError
            If data is missing or dimensions don't match
        """
        if self.real_signal is None or self.generated_signal is None:
            raise ValueError("Both real and generated signal data must be provided.")
            
        if self.real_signal.shape != self.generated_signal.shape:
            raise ValueError(
                f"Shape mismatch: real signal {self.real_signal.shape} vs "
                f"generated signal {self.generated_signal.shape}"
            )
    
    def cross_correlation(self, max_lag: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute cross-correlation between real and generated signals.
        
        Parameters
        ----------
        max_lag : int, optional
            Maximum lag to consider
            
        Returns
        -------
        lags : ndarray
            Lag values
        xcorr : ndarray, shape (n_channels, 2*max_lag+1)
            Cross-correlation for each channel
        """
        self._validate_data()
            
        n_channels = self.real_signal.shape[0]
        lags = np.arange(-max_lag, max_lag+1)
        xcorr = np.zeros((n_channels, len(lags)))
        
        for ch in range(n_channels):
            # Get normalized cross-correlation
            correlation = self._compute_cross_correlation(
                self.real_signal[ch, :],
                self.generated_signal[ch, :],
                max_lag
            )
            xcorr[ch, :] = correlation
            
        return lags, xcorr
    
    def _compute_cross_correlation(self, signal1: np.ndarray, signal2: np.ndarray, 
                                  max_lag: int) -> np.ndarray:
        """
        Compute normalized cross-correlation between two signals.
        
        Parameters
        ----------
        signal1, signal2 : ndarray
            Input signals
        max_lag : int
            Maximum lag to consider
            
        Returns
        -------
        xcorr : ndarray
            Normalized cross-correlation
        """
        # Use FFT-based cross-correlation (much faster for long signals)
        full_xcorr = signal.correlate(signal1, signal2, mode='full', method='fft')
        
        # Extract relevant portion around the center
        mid_point = len(full_xcorr) // 2
        xcorr = full_xcorr[mid_point-max_lag:mid_point+max_lag+1]
        
        # Normalize
        norm_factor = np.sqrt(np.sum(signal1**2) * np.sum(signal2**2))
        if norm_factor > 0:
            xcorr /= norm_factor
            
        return xcorr
        
    def mutual_information(self, bins: int = 10) -> np.ndarray:
        """
        Compute mutual information between real and generated signals.
        
        Parameters
        ----------
        bins : int, optional
            Number of bins for histogram
            
        Returns
        -------
        mi : ndarray, shape (n_channels,)
            Mutual information for each channel
        """
        self._validate_data()
            
        n_channels = self.real_signal.shape[0]
        mi = np.zeros(n_channels)
        
        for ch in range(n_channels):
            mi[ch] = self._compute_mutual_information(
                self.real_signal[ch, :],
                self.generated_signal[ch, :],
                bins
            )
                    
        return mi
    
    def _compute_mutual_information(self, signal1: np.ndarray, signal2: np.ndarray, 
                                   bins: int) -> float:
        """
        Compute mutual information between two signals using binning.
        
        Parameters
        ----------
        signal1, signal2 : ndarray
            Input signals
        bins : int
            Number of bins for histogram
            
        Returns
        -------
        mi : float
            Mutual information value
        """
        # Compute histograms
        hist_1, bin_edges = np.histogram(signal1, bins=bins)
        hist_2, _ = np.histogram(signal2, bins=bin_edges)
        
        hist_joint, _, _ = np.histogram2d(
            signal1, 
            signal2, 
            bins=[bin_edges, bin_edges]
        )
        
        # Normalize to get probability distributions
        p_1 = hist_1 / np.sum(hist_1)
        p_2 = hist_2 / np.sum(hist_2)
        p_joint = hist_joint / np.sum(hist_joint)
        
        # Compute mutual information
        mi = 0
        for i in range(bins):
            for j in range(bins):
                if p_joint[i, j] > 0 and p_1[i] > 0 and p_2[j] > 0:
                    mi += p_joint[i, j] * np.log2(p_joint[i, j] / (p_1[i] * p_2[j]))
                    
        return mi
        
    def network_reconstruction(self, method: str = 'correlation', 
                               threshold: Optional[float] = 0.3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reconstruct network connectivity from signal data.
        
        Parameters
        ----------
        method : str, optional
            Method for reconstruction:
            - 'correlation': Pearson correlation
            - 'mutual_info': Mutual information
            - 'coherence': Spectral coherence
        threshold : float, optional
            Threshold for binarization
            
        Returns
        -------
        real_network, gen_network : ndarray, shape (n_channels, n_channels)
            Reconstructed networks for real and generated signals
        """
        self._validate_data()
            
        n_channels = self.real_signal.shape[0]
        
        # Initialize connectivity matrices
        real_network = np.zeros((n_channels, n_channels))
        gen_network = np.zeros((n_channels, n_channels))
        
        # Select reconstruction method
        if method == 'correlation':
            real_network = self._correlation_matrix(self.real_signal)
            gen_network = self._correlation_matrix(self.generated_signal)
                    
        elif method == 'mutual_info':
            real_network = self._mutual_info_matrix(self.real_signal)
            gen_network = self._mutual_info_matrix(self.generated_signal)
                    
        elif method == 'coherence':
            real_network = self._coherence_matrix(self.real_signal)
            gen_network = self._coherence_matrix(self.generated_signal)
        
        else:
            raise ValueError(f"Unknown reconstruction method: {method}")
        
        # Apply threshold if needed
        if threshold is not None:
            real_network = (real_network > threshold).astype(float)
            gen_network = (gen_network > threshold).astype(float)
            
        return real_network, gen_network
    
    def _correlation_matrix(self, data: np.ndarray) -> np.ndarray:
        """
        Compute correlation matrix for multi-channel data.
        
        Parameters
        ----------
        data : ndarray, shape (n_channels, n_samples)
            Multi-channel data
            
        Returns
        -------
        corr_matrix : ndarray, shape (n_channels, n_channels)
            Correlation matrix
        """
        n_channels = data.shape[0]
        corr_matrix = np.zeros((n_channels, n_channels))
        
        # Compute full correlation matrix in one go
        for i in range(n_channels):
            for j in range(i+1, n_channels):  # Only compute upper triangle
                corr = np.corrcoef(data[i, :], data[j, :])[0, 1]
                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr  # Matrix is symmetric
                
        return corr_matrix
    
    def _mutual_info_matrix(self, data: np.ndarray, bins: int = 10) -> np.ndarray:
        """
        Compute mutual information matrix for multi-channel data.
        
        Parameters
        ----------
        data : ndarray, shape (n_channels, n_samples)
            Multi-channel data
        bins : int, optional
            Number of bins for histogram
            
        Returns
        -------
        mi_matrix : ndarray, shape (n_channels, n_channels)
            Mutual information matrix
        """
        n_channels = data.shape[0]
        mi_matrix = np.zeros((n_channels, n_channels))
        
        for i in range(n_channels):
            for j in range(i+1, n_channels):  # Only compute upper triangle
                mi = self._compute_mutual_information(data[i, :], data[j, :], bins)
                mi_matrix[i, j] = mi
                mi_matrix[j, i] = mi  # Matrix is symmetric
                
        return mi_matrix
    
    def _coherence_matrix(self, data: np.ndarray) -> np.ndarray:
        """
        Compute coherence matrix for multi-channel data.
        
        Parameters
        ----------
        data : ndarray, shape (n_channels, n_samples)
            Multi-channel data
            
        Returns
        -------
        coh_matrix : ndarray, shape (n_channels, n_channels)
            Coherence matrix (mean coherence across frequencies)
        """
        n_channels = data.shape[0]
        coh_matrix = np.zeros((n_channels, n_channels))
        
        for i in range(n_channels):
            for j in range(i+1, n_channels):  # Only compute upper triangle
                f, coh = signal.coherence(data[i, :], data[j, :])
                coh_matrix[i, j] = np.mean(coh)
                coh_matrix[j, i] = coh_matrix[i, j]  # Matrix is symmetric
                
        return coh_matrix
        
    def evaluate_reconstruction(self, real_network: np.ndarray, 
                               gen_network: np.ndarray) -> Dict[str, float]:
        """
        Evaluate network reconstruction performance.
        
        Parameters
        ----------
        real_network, gen_network : ndarray, shape (n_channels, n_channels)
            Reconstructed networks
            
        Returns
        -------
        metrics : dict
            Performance metrics (TPR, FPR, accuracy, etc.)
        """
        # Ensure networks are binary
        real_bin = (real_network > 0).astype(int)
        gen_bin = (gen_network > 0).astype(int)
        
        # Compute confusion matrix elements
        np.fill_diagonal(real_bin, 0)  # Ignore self-connections
        np.fill_diagonal(gen_bin, 0)
        
        TP = np.sum((real_bin == 1) & (gen_bin == 1))
        FP = np.sum((real_bin == 0) & (gen_bin == 1))
        FN = np.sum((real_bin == 1) & (gen_bin == 0))
        TN = np.sum((real_bin == 0) & (gen_bin == 0))
        
        # Calculate metrics
        TPR = TP / (TP + FN) if (TP + FN) > 0 else 0  # True Positive Rate (Sensitivity)
        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0  # False Positive Rate
        accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        f1_score = 2 * precision * TPR / (precision + TPR) if (precision + TPR) > 0 else 0
        
        metrics = {
            'TP': TP,
            'FP': FP,
            'FN': FN,
            'TN': TN,
            'TPR': TPR,
            'FPR': FPR,
            'accuracy': accuracy,
            'precision': precision,
            'f1_score': f1_score
        }
        
        return metrics
    
    def update_model_parameters(self, oscillator_network, learning_rate: float = 0.1):
        """
        Update oscillator network parameters based on statistical analysis.
        
        Parameters
        ----------
        oscillator_network : OscillatorNetwork
            Model to update
        learning_rate : float, optional
            Learning rate for updates
            
        Returns
        -------
        updated_model : OscillatorNetwork
            Model with updated parameters
        """
        self._validate_data()
        
        # Create a copy of the model to avoid modifying the original
        updated_model = oscillator_network
        
        # 1. Update adjacency matrix based on network reconstruction
        self._update_adjacency_matrix(updated_model, learning_rate)
        
        # 2. Update natural frequencies based on spectral properties
        self._update_frequencies(updated_model, learning_rate)
        
        # 3. Adjust global coupling strength based on synchronization differences
        self._update_coupling_strength(updated_model, learning_rate)
        
        return updated_model
    
    def _update_adjacency_matrix(self, model, learning_rate: float) -> None:
        """
        Update model's adjacency matrix based on network comparison.
        
        Parameters
        ----------
        model : OscillatorNetwork
            Model to update
        learning_rate : float
            Learning rate for updates
        """
        # Reconstruct networks
        real_network, gen_network = self.network_reconstruction(method='correlation')
        
        # Compute error matrix
        error_matrix = real_network - gen_network
        
        # Update adjacency matrix
        model.adjacency_matrix += learning_rate * error_matrix
        
        # Ensure non-negativity and constraint values
        model.adjacency_matrix = np.clip(model.adjacency_matrix, 0, 1)
        np.fill_diagonal(model.adjacency_matrix, 0)  # No self-connections
    
    def _update_frequencies(self, model, learning_rate: float) -> None:
        """
        Update model's natural frequencies based on spectral properties.
        
        Parameters
        ----------
        model : OscillatorNetwork
            Model to update
        learning_rate : float
            Learning rate for updates
        """
        # Calculate peak frequencies from PSD
        freqs_real, psd_real = self.compute_psd(self.real_signal)
        freqs_gen, psd_gen = self.compute_psd(self.generated_signal)
        
        # Find peak frequencies (weighted average approach)
        peak_freqs_real = self._calculate_weighted_peak_freq(freqs_real, psd_real)
        peak_freqs_gen = self._calculate_weighted_peak_freq(freqs_gen, psd_gen)
        
        # Update frequencies to reduce difference
        freq_diff = peak_freqs_real - peak_freqs_gen
        model.frequencies += learning_rate * freq_diff
    
    def _calculate_weighted_peak_freq(self, freqs: np.ndarray, psd: np.ndarray) -> np.ndarray:
        """
        Calculate weighted peak frequency for each channel.
        
        Parameters
        ----------
        freqs : ndarray
            Frequency bins
        psd : ndarray, shape (n_channels, n_freqs)
            Power spectral density
            
        Returns
        -------
        peak_freqs : ndarray, shape (n_channels,)
            Weighted peak frequency for each channel
        """
        n_channels = psd.shape[0]
        peak_freqs = np.zeros(n_channels)
        
        for ch in range(n_channels):
            # Use weighted average of frequencies, weighted by power
            peak_freqs[ch] = np.sum(freqs * psd[ch, :]) / np.sum(psd[ch, :]) if np.sum(psd[ch, :]) > 0 else 0
            
        return peak_freqs
    
    def _update_coupling_strength(self, model, learning_rate: float) -> None:
        """
        Update model's global coupling strength based on synchronization differences.
        
        Parameters
        ----------
        model : OscillatorNetwork
            Model to update
        learning_rate : float
            Learning rate for updates
        """
        # Extract phases from real and generated signals
        _, real_phases = self._extract_phases(self.real_signal)
        _, gen_phases = self._extract_phases(self.generated_signal)
        
        # Calculate order parameters
        real_order = self._calculate_order_parameter(real_phases)
        gen_order = self._calculate_order_parameter(gen_phases)
        
        # Update coupling strength based on synchronization difference
        sync_diff = real_order - gen_order
        model.global_coupling += learning_rate * sync_diff
        
        # Ensure positive coupling
        model.global_coupling = max(0.1, model.global_coupling)
    
    def _extract_phases(self, signal_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract analytic signal and instantaneous phases from signal data.
        
        Parameters
        ----------
        signal_data : ndarray, shape (n_channels, n_samples)
            Signal data
            
        Returns
        -------
        analytic_signal : ndarray, shape (n_channels, n_samples)
            Analytic signal
        phases : ndarray, shape (n_channels, n_samples)
            Instantaneous phases
        """
        n_channels, n_samples = signal_data.shape
        analytic_signal = np.zeros((n_channels, n_samples), dtype=complex)
        phases = np.zeros((n_channels, n_samples))
        
        for ch in range(n_channels):
            analytic_signal[ch, :] = signal.hilbert(signal_data[ch, :])
            phases[ch, :] = np.angle(analytic_signal[ch, :])
            
        return analytic_signal, phases
    
    def _calculate_order_parameter(self, phases: np.ndarray) -> float:
        """
        Calculate Kuramoto order parameter from phases.
        
        Parameters
        ----------
        phases : ndarray, shape (n_channels, n_samples)
            Phase time series
            
        Returns
        -------
        order_param : float
            Mean order parameter
        """
        # Transpose to get shape (n_samples, n_channels)
        phases_t = phases.T
        
        # Calculate complex order parameter for each time point
        z = np.mean(np.exp(1j * phases_t), axis=1)
        
        # Return mean absolute value
        return np.mean(np.abs(z))
    
    def compute_psd(self, signal_data: np.ndarray, fs: float = 256, 
                   fmin: float = 1, fmax: float = 45, 
                   n_fft: int = 512) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Power Spectral Density.
        
        Parameters
        ----------
        signal_data : ndarray, shape (n_channels, n_time_points)
            Signal data
        fs : float, optional
            Sampling frequency
        fmin, fmax : float, optional
            Minimum and maximum frequencies to consider
        n_fft : int, optional
            Number of FFT points (target)
            
        Returns
        -------
        freqs : ndarray
            Frequency bins
        psd : ndarray, shape (n_channels, n_freqs)
            Power spectral density for each channel
        """
        n_channels, n_samples = signal_data.shape
        
        # Adjust nperseg if signal is shorter than n_fft
        nperseg = min(n_fft, n_samples)
        # Ensure noverlap is valid (must be < nperseg)
        noverlap = nperseg // 2
        
        # Compute PSD for first channel to determine frequencies and shape
        f, Pxx_first = signal.welch(signal_data[0, :], fs=fs, nperseg=nperseg, noverlap=noverlap)
        
        # Create mask
        mask = (f >= fmin) & (f <= fmax)
        freqs = f[mask]
        
        # Initialize Output
        n_freq_bins = len(freqs)
        psd = np.zeros((n_channels, n_freq_bins))
        
        # Fill first channel
        psd[0, :] = Pxx_first[mask]
        
        # Compute for remaining channels
        for ch in range(1, n_channels):
            _, Pxx = signal.welch(signal_data[ch, :], fs=fs, nperseg=nperseg, noverlap=noverlap)
            psd[ch, :] = Pxx[mask]
            
        return freqs, psd