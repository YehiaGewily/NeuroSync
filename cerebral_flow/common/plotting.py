"""
Plotting Utilities Module
========================

This module provides visualization functions for signal data, oscillator network parameters,
and analysis results.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import networkx as nx
from mpl_toolkits.mplot3d import Axes3D


def plot_signal_comparison(real_signal, generated_signal, time_range=None, channels=None):
    """
    Plot comparison between real and generated signals.
    
    Parameters:
    -----------
    real_signal : ndarray, shape (n_channels, n_time_points)
        Real signal data
    generated_signal : ndarray, shape (n_channels, n_time_points)
        Generated signal data
    time_range : tuple, optional
        Range of time points to plot (start, end)
    channels : list, optional
        Channels to plot (if None, plots first channel)
    """
    if time_range is None:
        time_range = (0, min(300, real_signal.shape[1]))
        
    if channels is None:
        channels = [0]  # Default to first channel
        
    n_channels = len(channels)
    
    plt.figure(figsize=(12, 3 * n_channels))
    
    for i, ch in enumerate(channels):
        plt.subplot(n_channels, 1, i + 1)
        t = np.arange(time_range[0], time_range[1])
        plt.plot(t, real_signal[ch, time_range[0]:time_range[1]], 'b-', label='Real Signal')
        plt.plot(t, generated_signal[ch, time_range[0]:time_range[1]], 'r-', label='Generated Signal')
        plt.title(f'Channel {ch}')
        plt.xlabel('Time (samples)')
        plt.ylabel('Amplitude')
        plt.legend()
        
    plt.tight_layout()
    
    
def plot_psd_comparison(real_signal, generated_signal, fs=256, fmax=50, channel=0):
    """
    Plot Power Spectral Density comparison.
    
    Parameters:
    -----------
    real_signal : ndarray, shape (n_channels, n_time_points)
        Real signal data
    generated_signal : ndarray, shape (n_channels, n_time_points)
        Generated signal data
    fs : float, optional
        Sampling frequency
    fmax : float, optional
        Maximum frequency to plot
    channel : int, optional
        Channel to plot
    """
    plt.figure(figsize=(10, 6))
    
    # Compute PSD for real signal
    f_real, Pxx_real = signal.welch(real_signal[channel], fs=fs, nperseg=512)
    mask_real = f_real <= fmax
    
    # Compute PSD for generated signal
    f_gen, Pxx_gen = signal.welch(generated_signal[channel], fs=fs, nperseg=512)
    mask_gen = f_gen <= fmax
    
    # Plot
    plt.semilogy(f_real[mask_real], Pxx_real[mask_real], 'b-', label='Real Signal')
    plt.semilogy(f_gen[mask_gen], Pxx_gen[mask_gen], 'r-', label='Generated Signal')
    
    plt.title(f'Power Spectral Density (Channel {channel})')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD (µV²/Hz)')
    plt.legend()
    plt.grid(True)
    
    
def plot_phases(phases, nodes=None, time_range=None):
    """
    Plot phase time series.
    
    Parameters:
    -----------
    phases : ndarray, shape (n_time_points, n_nodes)
        Phase time series
    nodes : list, optional
        Nodes to plot (if None, plots all)
    time_range : tuple, optional
        Range of time points to plot (start, end)
    """
    if time_range is None:
        time_range = (0, phases.shape[0])
        
    if nodes is None:
        nodes = np.arange(phases.shape[1])
        
    plt.figure(figsize=(10, 6))
    
    t = np.arange(time_range[0], time_range[1])
    
    for node in nodes:
        plt.plot(t, phases[time_range[0]:time_range[1], node], label=f'Node {node}')
        
    plt.title('Phase Evolution')
    plt.xlabel('Time')
    plt.ylabel('Phase (rad)')
    plt.legend()
    plt.grid(True)
    
    
def plot_order_parameter(times, order_parameter):
    """
    Plot Kuramoto order parameter.
    
    Parameters:
    -----------
    times : ndarray
        Time points
    order_parameter : ndarray
        Order parameter values
    """
    plt.figure(figsize=(10, 6))
    
    plt.plot(times, order_parameter)
    plt.title('Order Parameter')
    plt.xlabel('Time')
    plt.ylabel('Order Parameter')
    plt.ylim(0, 1)
    plt.grid(True)
    
    
def plot_connectivity_matrix(matrix, title='Connectivity Matrix'):
    """
    Plot connectivity matrix as a heatmap.
    
    Parameters:
    -----------
    matrix : ndarray, shape (n, n)
        Connectivity matrix
    title : str, optional
        Plot title
    """
    plt.figure(figsize=(8, 6))
    
    plt.imshow(matrix, cmap='viridis', interpolation='none')
    plt.colorbar(label='Connection Strength')
    plt.title(title)
    plt.xlabel('Node')
    plt.ylabel('Node')
    
    
def plot_network_graph(matrix, threshold=0.1, node_size=300, node_color='skyblue'):
    """
    Plot network graph from connectivity matrix.
    
    Parameters:
    -----------
    matrix : ndarray, shape (n, n)
        Connectivity matrix
    threshold : float, optional
        Threshold for including edges
    node_size : float, optional
        Size of nodes in plot
    node_color : str, optional
        Color of nodes
    """
    # Create binary adjacency matrix
    binary_matrix = (matrix > threshold).astype(int)
    
    # Create graph
    G = nx.from_numpy_array(binary_matrix)
    
    # Calculate node positions using spring layout
    pos = nx.spring_layout(G)
    
    plt.figure(figsize=(10, 8))
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_color, alpha=0.8)
    
    # Draw edges with width proportional to weight
    for (u, v, d) in G.edges(data=True):
        weight = matrix[u, v]
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=weight*3, alpha=0.7)
    
    # Draw node labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
    
    plt.axis('off')
    plt.title('Network Graph')
    
    
def plot_error_history(error_history):
    """
    Plot error history from refinement process.
    
    Parameters:
    -----------
    error_history : list or ndarray
        Error values at each iteration
    """
    plt.figure(figsize=(10, 6))
    
    plt.plot(np.arange(1, len(error_history)+1), error_history, 'b-o')
    plt.title('Error History')
    plt.xlabel('Iteration')
    plt.ylabel('Mean Absolute Error')
    plt.grid(True)
    
    
def plot_parameter_evolution(parameter_history):
    """
    Plot evolution of parameters during refinement.
    
    Parameters:
    -----------
    parameter_history : dict
        Dictionary with parameter history
    """
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Coupling strength
    plt.subplot(3, 1, 1)
    plt.plot(np.arange(1, len(parameter_history['coupling'])+1), parameter_history['coupling'], 'b-o')
    plt.title('Global Coupling Strength')
    plt.xlabel('Iteration')
    plt.ylabel('Coupling')
    plt.grid(True)
    
    # Plot 2: Frequencies
    plt.subplot(3, 1, 2)
    freqs = np.array(parameter_history['frequencies'])
    for i in range(freqs.shape[1]):
        plt.plot(np.arange(1, freqs.shape[0]+1), freqs[:, i], label=f'Node {i}')
    plt.title('Natural Frequencies')
    plt.xlabel('Iteration')
    plt.ylabel('Frequency')
    plt.grid(True)
    if freqs.shape[1] <= 10:  # Only show legend if not too many nodes
        plt.legend()
    
    # Plot 3: Connectivity matrix norm
    plt.subplot(3, 1, 3)
    conn_norm = [np.linalg.norm(conn) for conn in parameter_history['connectivity']]
    plt.plot(np.arange(1, len(conn_norm)+1), conn_norm, 'g-o')
    plt.title('Connectivity Matrix Norm')
    plt.xlabel('Iteration')
    plt.ylabel('Frobenius Norm')
    plt.grid(True)
    
    plt.tight_layout()
    
    
def plot_phase_space(phases, velocities=None, nodes=None):
    """
    Plot phase space trajectory for inertial oscillator model.
    
    Parameters:
    -----------
    phases : ndarray, shape (n_time_points, n_nodes)
        Phase time series
    velocities : ndarray, shape (n_time_points, n_nodes), optional
        Velocity time series (for inertial model)
    nodes : list, optional
        Nodes to plot (if None, plots first node)
    """
    if nodes is None:
        nodes = [0]  # Default to first node
        
    if velocities is None:
        # Standard model: plot phases and their derivatives
        plt.figure(figsize=(10, 6))
        
        for node in nodes:
            phase = phases[:, node]
            # Estimate derivative
            dphase = np.diff(phase)
            
            plt.plot(phase[:-1], dphase, 'o-', label=f'Node {node}')
            
        plt.title('Phase Space (Phase vs. Phase Velocity)')
        plt.xlabel('Phase')
        plt.ylabel('Phase Velocity')
        plt.legend()
        plt.grid(True)
        
    else:
        # Inertial model: plot 3D phase space
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        for node in nodes:
            phase = phases[:, node]
            velocity = velocities[:, node]
            # Estimate acceleration
            accel = np.diff(velocity)
            
            ax.plot(phase[:-1], velocity[:-1], accel, 'o-', label=f'Node {node}')
            
        ax.set_title('Phase Space (Phase, Velocity, Acceleration)')
        ax.set_xlabel('Phase')
        ax.set_ylabel('Velocity')
        ax.set_zlabel('Acceleration')
        plt.legend()
        
        
def plot_arnold_tongues(freq_grid, coupling_grid, sync_grid):
    """
    Plot Arnold tongues from parameter sweep.
    
    Parameters:
    -----------
    freq_grid, coupling_grid : ndarray
        Meshgrid of frequency and coupling values
    sync_grid : ndarray
        Synchronization level for each parameter combination
    """
    plt.figure(figsize=(10, 8))
    
    plt.pcolormesh(freq_grid, coupling_grid, sync_grid, cmap='viridis', shading='auto')
    plt.colorbar(label='Synchronization')
    
    plt.title('Arnold Tongues')
    plt.xlabel('Frequency Detuning')
    plt.ylabel('Coupling Strength')
    
    plt.tight_layout()
    
    
def plot_full_results(real_signal, generated_signal, oscillator_network, error_history):
    """
    Create a comprehensive results figure.
    
    Parameters:
    -----------
    real_signal : ndarray
        Real signal data
    generated_signal : ndarray
        Generated signal data
    oscillator_network : OscillatorNetwork or DynamicOscillatorNetwork
        Oscillator network model
    error_history : list or ndarray
        Error history from refinement
    """
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Compare real vs generated signal for one channel
    plt.subplot(3, 2, 1)
    ch = 0  # First channel
    t = np.arange(min(300, real_signal.shape[1]))  # Show first 300 samples or fewer
    plt.plot(t, real_signal[ch, :len(t)], 'b-', label='Real Signal')
    plt.plot(t, generated_signal[ch, :len(t)], 'r-', label='Generated Signal')
    plt.title('Real vs Generated Signal (Channel 1)')
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')
    plt.legend()
    
    # Plot 2: Error history
    plt.subplot(3, 2, 2)
    plt.plot(np.arange(1, len(error_history)+1), error_history, 'b-o')
    plt.title('Error History')
    plt.xlabel('Iteration')
    plt.ylabel('Mean Absolute Error')
    plt.grid(True)
    
    # Plot 3: Power spectral density
    plt.subplot(3, 2, 3)
    fs = 256  # Sampling frequency (example)
    f_real, Pxx_real = signal.welch(real_signal[ch], fs=fs, nperseg=512)
    f_gen, Pxx_gen = signal.welch(generated_signal[ch], fs=fs, nperseg=512)
    plt.semilogy(f_real, Pxx_real, 'b-', label='Real Signal')
    plt.semilogy(f_gen, Pxx_gen, 'r-', label='Generated Signal')
    plt.title('Power Spectral Density (Channel 1)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD (µV²/Hz)')
    plt.legend()
    plt.xlim([0, 50])  # Show frequencies up to 50 Hz
    
    # Plot 4: Oscillator network frequencies
    plt.subplot(3, 2, 4)
    plt.hist(oscillator_network.frequencies, bins=10)
    plt.title('Distribution of Natural Frequencies')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Count')
    
    # Plot 5: Oscillator network connectivity
    plt.subplot(3, 2, 5)
    plt.imshow(oscillator_network.adjacency_matrix, cmap='viridis', interpolation='none')
    plt.colorbar(label='Connection Strength')
    plt.title('Oscillator Network Connectivity Matrix')
    plt.xlabel('Node')
    plt.ylabel('Node')
    
    # Plot 6: Order parameter
    plt.subplot(3, 2, 6)
    _, phases, order_param = oscillator_network.simulate(duration=5.0, dt=0.01)
    t = np.arange(0, 5.0, 0.01)
    plt.plot(t, order_param)
    plt.title('Order Parameter (Synchronization Level)')
    plt.xlabel('Time (s)')
    plt.ylabel('Order Parameter')
    plt.ylim([0, 1])
    plt.grid(True)
    
    plt.tight_layout()