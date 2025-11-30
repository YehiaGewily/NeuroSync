"""
Common Utilities Package
=======================

This package contains common utility modules for metrics and plotting.
"""

from .metrics import (
    mean_absolute_error,
    pearson_correlation,
    cross_correlation_max,
    spectral_coherence,
    mutual_information_score,
    order_parameter,
    network_metrics,
    compare_networks
)

from .plotting import (
    plot_signal_comparison,
    plot_psd_comparison,
    plot_phases,
    plot_order_parameter,
    plot_connectivity_matrix,
    plot_network_graph,
    plot_error_history,
    plot_parameter_evolution,
    plot_phase_space,
    plot_arnold_tongues,
    plot_full_results
)

__all__ = [
    'mean_absolute_error',
    'pearson_correlation',
    'cross_correlation_max',
    'spectral_coherence',
    'mutual_information_score',
    'order_parameter',
    'network_metrics',
    'compare_networks',
    'plot_signal_comparison',
    'plot_psd_comparison',
    'plot_phases',
    'plot_order_parameter',
    'plot_connectivity_matrix',
    'plot_network_graph',
    'plot_error_history',
    'plot_parameter_evolution',
    'plot_phase_space',
    'plot_arnold_tongues',
    'plot_full_results'
]