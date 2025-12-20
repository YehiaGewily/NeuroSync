"""
Advanced Connectivity Metrics Module
===================================

This module implements robust functional connectivity metrics that are less
sensitive to volume conduction artifacts.
"""

import numpy as np
from scipy import signal

def phase_lag_index(phases):
    """
    Calculate Phase Lag Index (PLI).
    
    PLI measures the asymmetry of the distribution of phase differences.
    It is insensitive to zero-lag synchrony (often due to volume conduction).
    
    Parameters:
    -----------
    phases : ndarray, shape (n_time_points, n_channels)
        Instantaneous phases
        
    Returns:
    --------
    pli_matrix : ndarray, shape (n_channels, n_channels)
        PLI connectivity matrix
    """
    n_time, n_channels = phases.shape
    pli_matrix = np.zeros((n_channels, n_channels))
    
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            # Phase difference
            d_phi = phases[:, i] - phases[:, j]
            
            # wrap to [-pi, pi]
            d_phi = (d_phi + np.pi) % (2 * np.pi) - np.pi
            
            # PLI = |mean(sign(d_phi))|
            # Note: Strict inequality implies ignoring exact zero/pi differences
            # For practical data, simple sign is usually sufficient
            pli = np.abs(np.mean(np.sign(np.sin(d_phi))))
            
            pli_matrix[i, j] = pli
            pli_matrix[j, i] = pli
            
    return pli_matrix

def weighted_phase_lag_index(phases):
    """
    Calculate Weighted Phase Lag Index (wPLI).
    
    wPLI weights the phase differences by the magnitude of the imaginary component
    of the cross-spectrum, mitigating the discontinuity of PLI around zero.
    
    Parameters:
    -----------
    phases : ndarray, shape (n_time_points, n_channels)
        Instantaneous phases
        
    Returns:
    --------
    wpli_matrix : ndarray, shape (n_channels, n_channels)
        wPLI connectivity matrix
    """
    n_time, n_channels = phases.shape
    wpli_matrix = np.zeros((n_channels, n_channels))
    
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            d_phi = phases[:, i] - phases[:, j]
            
            # Imaginary component of cross-spectrum ~ sin(d_phi)
            im_cross = np.sin(d_phi)
            
            num = np.abs(np.mean(im_cross))
            denom = np.mean(np.abs(im_cross))
            
            wpli = num / denom if denom > 1e-10 else 0
            
            wpli_matrix[i, j] = wpli
            wpli_matrix[j, i] = wpli
            
    return wpli_matrix

def amplitude_envelope_correlation(signal_data, window_size=None):
    """
    Calculate Amplitude Envelope Correlation (AEC).
    
    Parameters:
    -----------
    signal_data : ndarray, shape (n_channels, n_samples)
    
    Returns:
    --------
    aec_matrix : ndarray
    """
    n_channels, n_samples = signal_data.shape
    
    # Hilbert transform to get envelope
    envelopes = np.abs(signal.hilbert(signal_data, axis=1))
    
    # Orthogonalization helps remove volume conduction (Hipp et al. 2012)
    # Here we implement standard AEC for simplicity, orthogonalization can be an extension
    
    aec_matrix = np.corrcoef(envelopes)
    # Remove diagonal
    np.fill_diagonal(aec_matrix, 0)
    
    return aec_matrix
