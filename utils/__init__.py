"""
Utility modules for the EEG-Kuramoto Neural Dynamics Framework.
"""

# Import main functions for easier access
from .plotting import (
    plot_eeg_comparison, plot_psd_comparison, plot_phases, 
    plot_order_parameter, plot_connectivity_matrix, plot_network_graph,
    plot_error_history, plot_parameter_evolution, plot_full_results
)

from .metrics import (
    mean_absolute_error, pearson_correlation, cross_correlation_max,
    spectral_coherence, mutual_information_score, order_parameter,
    network_metrics, compare_networks
)

__all__ = [
    'plot_eeg_comparison', 'plot_psd_comparison', 'plot_phases', 
    'plot_order_parameter', 'plot_connectivity_matrix', 'plot_network_graph',
    'plot_error_history', 'plot_parameter_evolution', 'plot_full_results',
    'mean_absolute_error', 'pearson_correlation', 'cross_correlation_max',
    'spectral_coherence', 'mutual_information_score', 'order_parameter',
    'network_metrics', 'compare_networks'
]