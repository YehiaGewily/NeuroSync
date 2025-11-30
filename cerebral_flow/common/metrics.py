"""
Metrics Utility Module
=====================

This module provides metrics for evaluating model performance and
comparing real and generated signal data.
"""

import numpy as np
from scipy import signal, stats


def mean_absolute_error(real_signal, generated_signal, normalize=True):
    """
    Calculate Mean Absolute Error between real and generated signals.
    
    Parameters:
    -----------
    real_signal : ndarray
        Real signal data
    generated_signal : ndarray
        Generated signal data
    normalize : bool, optional
        Whether to normalize data before computing MAE
        
    Returns:
    --------
    mae : float
        Mean Absolute Error
    """
    if normalize:
        real_norm = (real_signal - np.mean(real_signal)) / np.std(real_signal)
        gen_norm = (generated_signal - np.mean(generated_signal)) / np.std(generated_signal)
        return np.mean(np.abs(real_norm - gen_norm))
    else:
        return np.mean(np.abs(real_signal - generated_signal))


def pearson_correlation(real_signal, generated_signal):
    """
    Calculate Pearson correlation coefficient.
    
    Parameters:
    -----------
    real_signal : ndarray, shape (n,)
        Real signal data (1D array)
    generated_signal : ndarray, shape (n,)
        Generated signal data (1D array)
        
    Returns:
    --------
    corr : float
        Pearson correlation coefficient
    p_value : float
        p-value for the correlation
    """
    return stats.pearsonr(real_signal, generated_signal)


def cross_correlation_max(real_signal, generated_signal, max_lag=100):
    """
    Calculate maximum cross-correlation and corresponding lag.
    
    Parameters:
    -----------
    real_signal : ndarray, shape (n,)
        Real signal data (1D array)
    generated_signal : ndarray, shape (n,)
        Generated signal data (1D array)
    max_lag : int, optional
        Maximum lag to consider
        
    Returns:
    --------
    max_corr : float
        Maximum cross-correlation value
    lag : int
        Lag at which maximum occurs
    """
    # Compute cross-correlation
    n = len(real_signal)
    corr = signal.correlate(real_signal, generated_signal, mode='same') / n
    lags = np.arange(-max_lag, max_lag + 1)
    corr = corr[n//2 - max_lag:n//2 + max_lag + 1]
    
    # Normalize
    corr /= np.sqrt(np.mean(real_signal**2) * np.mean(generated_signal**2))
    
    # Find maximum
    max_idx = np.argmax(np.abs(corr))
    max_corr = corr[max_idx]
    lag = lags[max_idx]
    
    return max_corr, lag


def spectral_coherence(real_signal, generated_signal, fs=256, fmin=0, fmax=100):
    """
    Calculate spectral coherence between real and generated signals.
    
    Parameters:
    -----------
    real_signal : ndarray, shape (n,)
        Real signal data (1D array)
    generated_signal : ndarray, shape (n,)
        Generated signal data (1D array)
    fs : float, optional
        Sampling frequency
    fmin, fmax : float, optional
        Frequency range to consider
        
    Returns:
    --------
    freqs : ndarray
        Frequency bins
    coherence : ndarray
        Coherence values
    mean_coherence : float
        Mean coherence in the specified frequency range
    """
    # Compute coherence
    f, Cxy = signal.coherence(real_signal, generated_signal, fs=fs)
    
    # Filter frequencies
    mask = (f >= fmin) & (f <= fmax)
    freqs = f[mask]
    coherence = Cxy[mask]
    
    # Calculate mean coherence
    mean_coherence = np.mean(coherence)
    
    return freqs, coherence, mean_coherence


def mutual_information_score(real_signal, generated_signal, bins=10):
    """
    Calculate mutual information between real and generated signals.
    
    Parameters:
    -----------
    real_signal : ndarray, shape (n,)
        Real signal data (1D array)
    generated_signal : ndarray, shape (n,)
        Generated signal data (1D array)
    bins : int, optional
        Number of bins for histogram
        
    Returns:
    --------
    mi : float
        Mutual information value
    """
    # Compute histograms
    hist_real, bin_edges = np.histogram(real_signal, bins=bins)
    hist_gen, _ = np.histogram(generated_signal, bins=bin_edges)
    
    hist_joint, _, _ = np.histogram2d(
        real_signal, 
        generated_signal, 
        bins=[bin_edges, bin_edges]
    )
    
    # Normalize to get probability distributions
    p_real = hist_real / np.sum(hist_real)
    p_gen = hist_gen / np.sum(hist_gen)
    p_joint = hist_joint / np.sum(hist_joint)
    
    # Compute mutual information
    mi = 0
    for i in range(bins):
        for j in range(bins):
            if p_joint[i, j] > 0 and p_real[i] > 0 and p_gen[j] > 0:
                mi += p_joint[i, j] * np.log2(p_joint[i, j] / (p_real[i] * p_gen[j]))
                
    return mi


def order_parameter(phases):
    """
    Calculate Kuramoto order parameter.
    
    Parameters:
    -----------
    phases : ndarray, shape (n_nodes,) or (n_time_points, n_nodes)
        Phases of oscillators
        
    Returns:
    --------
    r : float or ndarray
        Order parameter (0 ≤ r ≤ 1)
    """
    if phases.ndim == 1:
        # Single time point
        z = np.mean(np.exp(1j * phases))
        return np.abs(z)
    else:
        # Multiple time points
        z = np.mean(np.exp(1j * phases), axis=1)
        return np.abs(z)


def network_metrics(adjacency_matrix, threshold=0.1):
    """
    Calculate network metrics from adjacency matrix.
    
    Parameters:
    -----------
    adjacency_matrix : ndarray, shape (n, n)
        Adjacency matrix
    threshold : float, optional
        Threshold for binarization
        
    Returns:
    --------
    metrics : dict
        Dictionary with network metrics
    """
    import networkx as nx
    
    # Create binary adjacency matrix
    binary_matrix = (adjacency_matrix > threshold).astype(int)
    
    # Create graph
    G = nx.from_numpy_array(binary_matrix)
    
    # Calculate metrics
    metrics = {
        'density': nx.density(G),
        'transitivity': nx.transitivity(G),
        'average_clustering': nx.average_clustering(G),
        'average_shortest_path_length': nx.average_shortest_path_length(G) if nx.is_connected(G) else np.nan,
        'assortativity': nx.degree_assortativity_coefficient(G),
        'number_of_nodes': G.number_of_nodes(),
        'number_of_edges': G.number_of_edges()
    }
    
    # Try to calculate metrics that might fail on disconnected graphs
    try:
        metrics['global_efficiency'] = nx.global_efficiency(G)
    except:
        metrics['global_efficiency'] = np.nan
        
    # Calculate node-level metrics
    node_degree = dict(G.degree())
    metrics['max_degree'] = max(node_degree.values()) if node_degree else 0
    metrics['min_degree'] = min(node_degree.values()) if node_degree else 0
    metrics['average_degree'] = np.mean(list(node_degree.values())) if node_degree else 0
    
    return metrics


def compare_networks(real_network, gen_network, threshold=0.1):
    """
    Compare network properties between real and generated networks.
    
    Parameters:
    -----------
    real_network, gen_network : ndarray, shape (n, n)
        Adjacency matrices for real and generated networks
    threshold : float, optional
        Threshold for binarization
        
    Returns:
    --------
    metrics : dict
        Dictionary with comparison metrics
    """
    # Get network metrics
    real_metrics = network_metrics(real_network, threshold)
    gen_metrics = network_metrics(gen_network, threshold)
    
    # Calculate differences
    diff_metrics = {}
    for key in real_metrics:
        if isinstance(real_metrics[key], (int, float)) and not np.isnan(real_metrics[key]) and not np.isnan(gen_metrics[key]):
            diff_metrics[f'{key}_difference'] = real_metrics[key] - gen_metrics[key]
            diff_metrics[f'{key}_relative_difference'] = (real_metrics[key] - gen_metrics[key]) / real_metrics[key] if real_metrics[key] != 0 else np.nan
    
    # Add some direct comparison metrics
    diff_metrics['adjacency_frobenius_norm'] = np.linalg.norm(real_network - gen_network)
    diff_metrics['adjacency_mean_error'] = np.mean(np.abs(real_network - gen_network))
    
    # Binarize networks
    real_bin = (real_network > threshold).astype(int)
    gen_bin = (gen_network > threshold).astype(int)
    
    # Calculate confusion matrix elements
    TP = np.sum((real_bin == 1) & (gen_bin == 1))
    FP = np.sum((real_bin == 0) & (gen_bin == 1))
    FN = np.sum((real_bin == 1) & (gen_bin == 0))
    TN = np.sum((real_bin == 0) & (gen_bin == 0))
    
    # Calculate standard metrics
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0  # Sensitivity
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0  # False Positive Rate
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    
    diff_metrics.update({
        'TP': TP,
        'FP': FP,
        'FN': FN,
        'TN': TN,
        'sensitivity': TPR,
        'false_positive_rate': FPR,
        'accuracy': accuracy,
        'precision': precision
    })
    
    return diff_metrics