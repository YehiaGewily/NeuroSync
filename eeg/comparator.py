"""
EEG Comparator Module
====================

This module compares generated and real EEG data, and implements functional control.
It corresponds to the seventh step in the framework.
"""

import numpy as np
from scipy import signal


class EEGComparator:
    """
    Compare generated and real EEG data, and implement functional control.
    
    This class provides methods to evaluate the similarity between real and generated
    EEG signals, and to adjust model parameters to better match the real data.
    """
    
    def __init__(self, real_eeg=None, generated_eeg=None):
        """
        Initialize comparator with real and generated EEG.
        
        Parameters:
        -----------
        real_eeg : ndarray, shape (n_channels, n_time_points), optional
            Real EEG data
        generated_eeg : ndarray, shape (n_channels, n_time_points), optional
            Generated EEG data
        """
        self.real_eeg = real_eeg
        self.generated_eeg = generated_eeg
        
    def set_data(self, real_eeg, generated_eeg):
        """
        Set the real and generated EEG data.
        
        Parameters:
        -----------
        real_eeg : ndarray, shape (n_channels, n_time_points)
            Real EEG data
        generated_eeg : ndarray, shape (n_channels, n_time_points)
            Generated EEG data
        """
        self.real_eeg = real_eeg
        self.generated_eeg = generated_eeg
        
    def phase_relationship_matrix(self, phases, time_window=None):
        """
        Compute phase relationship matrix from phases.
        
        Parameters:
        -----------
        phases : ndarray, shape (n_time_points, n_channels)
            Phase time series
        time_window : tuple, optional
            Time window to consider (start, end indices)
            
        Returns:
        --------
        relationship_matrix : ndarray, shape (n_channels, n_channels)
            Phase relationship matrix
        """
        if time_window is not None:
            start, end = time_window
            phases = phases[start:end, :]
            
        n_channels = phases.shape[1]
        relationship_matrix = np.zeros((n_channels, n_channels))
        
        for i in range(n_channels):
            for j in range(n_channels):
                if i != j:
                    # Average cosine of phase differences
                    phase_diff = phases[:, j] - phases[:, i]
                    relationship_matrix[i, j] = np.mean(np.cos(phase_diff))
                    
        return relationship_matrix
    
    def mean_absolute_error(self, time_window=None):
        """
        Calculate Mean Absolute Error between real and generated EEG.
        
        Parameters:
        -----------
        time_window : tuple, optional
            Time window to consider (start, end indices)
            
        Returns:
        --------
        mae : float
            Mean Absolute Error
            
        Raises:
        -------
        ValueError
            If either real or generated EEG data is missing
        """
        if self.real_eeg is None or self.generated_eeg is None:
            raise ValueError("Both real and generated EEG data must be provided.")
            
        if time_window is not None:
            start, end = time_window
            real = self.real_eeg[:, start:end]
            gen = self.generated_eeg[:, start:end]
        else:
            real = self.real_eeg
            gen = self.generated_eeg
            
        # Normalize data
        real_norm = (real - np.mean(real)) / np.std(real)
        gen_norm = (gen - np.mean(gen)) / np.std(gen)
        
        # Calculate MAE
        mae = np.mean(np.abs(real_norm - gen_norm))
        
        return mae
    
    def pearson_correlation(self, time_window=None):
        """
        Calculate Pearson correlation between real and generated EEG.
        
        Parameters:
        -----------
        time_window : tuple, optional
            Time window to consider (start, end indices)
            
        Returns:
        --------
        correlations : ndarray, shape (n_channels,)
            Correlation coefficient for each channel
            
        Raises:
        -------
        ValueError
            If either real or generated EEG data is missing
        """
        if self.real_eeg is None or self.generated_eeg is None:
            raise ValueError("Both real and generated EEG data must be provided.")
            
        if time_window is not None:
            start, end = time_window
            real = self.real_eeg[:, start:end]
            gen = self.generated_eeg[:, start:end]
        else:
            real = self.real_eeg
            gen = self.generated_eeg
            
        n_channels = real.shape[0]
        correlations = np.zeros(n_channels)
        
        for ch in range(n_channels):
            correlations[ch] = np.corrcoef(real[ch, :], gen[ch, :])[0, 1]
            
        return correlations
    
    def compare_psd(self, fs=256, fmin=1, fmax=45, n_fft=512):
        """
        Compare Power Spectral Density between real and generated EEG.
        
        Parameters:
        -----------
        fs, fmin, fmax, n_fft : parameters for PSD computation
            
        Returns:
        --------
        freqs : ndarray
            Frequency bins
        real_psd, gen_psd : ndarray, shape (n_channels, n_freqs)
            PSD for real and generated data
            
        Raises:
        -------
        ValueError
            If either real or generated EEG data is missing
        """
        if self.real_eeg is None or self.generated_eeg is None:
            raise ValueError("Both real and generated EEG data must be provided.")
            
        # Compute PSD for real EEG
        freqs = np.linspace(0, fs/2, n_fft//2 + 1)
        mask = (freqs >= fmin) & (freqs <= fmax)
        freqs = freqs[mask]
        
        n_channels = self.real_eeg.shape[0]
        real_psd = np.zeros((n_channels, np.sum(mask)))
        gen_psd = np.zeros((n_channels, np.sum(mask)))
        
        for ch in range(n_channels):
            # Real EEG
            f, Pxx_real = signal.welch(self.real_eeg[ch, :], fs=fs, nperseg=n_fft, noverlap=n_fft//2)
            real_psd[ch, :] = Pxx_real[mask]
            
            # Generated EEG
            f, Pxx_gen = signal.welch(self.generated_eeg[ch, :], fs=fs, nperseg=n_fft, noverlap=n_fft//2)
            gen_psd[ch, :] = Pxx_gen[mask]
            
        return freqs, real_psd, gen_psd
    
    def functional_control(self, kuramoto_model, target_pattern, learning_rate=0.01, max_iterations=100):
        """
        Implement functional control by adjusting Kuramoto parameters.
        
        Parameters:
        -----------
        kuramoto_model : KuramotoModel
            Model to control
        target_pattern : ndarray, shape (n_channels, n_channels)
            Target phase relationship pattern
        learning_rate : float, optional
            Learning rate for parameter updates
        max_iterations : int, optional
            Maximum number of iterations
            
        Returns:
        --------
        updated_model : KuramotoModel
            Model with updated parameters
        error_history : list
            Error at each iteration
        """
        # Make a copy of the model
        updated_model = kuramoto_model
        
        # Store original parameters
        original_coupling = updated_model.global_coupling
        original_adjacency = updated_model.adjacency_matrix.copy()
        
        # Initialize error history
        error_history = []
        
        # Iterative optimization
        for iteration in range(max_iterations):
            # Simulate with current parameters
            _, phases, _ = updated_model.simulate(duration=10.0, dt=0.1)
            
            # Compute current pattern
            current_pattern = self.phase_relationship_matrix(phases)
            
            # Compute error
            error = np.mean((current_pattern - target_pattern)**2)
            error_history.append(error)
            
            # Check convergence
            if error < 0.01:
                break
                
            # Update parameters
            # 1. Adjust coupling strength
            gradient = 2 * (current_pattern - target_pattern)
            updated_model.global_coupling -= learning_rate * np.mean(gradient)
            
            # 2. Adjust adjacency matrix (with constraints to maintain structure)
            for i in range(updated_model.n_oscillators):
                for j in range(updated_model.n_oscillators):
                    if original_adjacency[i, j] > 0:  # Only adjust existing connections
                        updated_model.adjacency_matrix[i, j] -= learning_rate * gradient[i, j]
                        # Ensure non-negativity
                        updated_model.adjacency_matrix[i, j] = max(0.1, updated_model.adjacency_matrix[i, j])
            
        return updated_model, error_history