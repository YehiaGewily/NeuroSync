"""
Example Pipeline Script
======================

This script demonstrates the use of the CerebralFlow framework
by simulating signal data and running the complete pipeline.
"""

import numpy as np
import matplotlib.pyplot as plt

# Import framework components
from cerebral_flow.signals.data_inversion import SignalInverter
from cerebral_flow.simulation.neural_mass import MassNeuralDynamics
from cerebral_flow.simulation.time_varying import DynamicOscillatorNetwork
from cerebral_flow.analytics.critical_sync import CriticalityAnalyzer
from cerebral_flow.signals.generator import SignalGenerator
from cerebral_flow.signals.comparator import SignalComparator
from cerebral_flow.analytics.statistics import StatisticalAnalyzer
from cerebral_flow.analytics.closed_loop import ClosedLoopOptimizer
from cerebral_flow.common.plotting import plot_full_results
from cerebral_flow.common.metrics import compare_networks


def generate_synthetic_signal(n_channels=19, duration=10, sampling_rate=256):
    """
    Generate synthetic signal data for demonstration.
    
    Parameters:
    -----------
    n_channels : int
        Number of channels
    duration : float
        Duration in seconds
    sampling_rate : float
        Sampling rate in Hz
        
    Returns:
    --------
    signal_data : ndarray, shape (n_channels, n_samples)
        Synthetic signal data
    """
    n_samples = int(duration * sampling_rate)
    
    # Create time vector
    t = np.arange(n_samples) / sampling_rate
    
    # Initialize signal data
    signal_data = np.zeros((n_channels, n_samples))
    
    for ch in range(n_channels):
        # Base frequency in alpha range (8-13 Hz) with small variations between channels
        base_freq = 10 + 0.5 * np.random.randn()
        
        # Generate alpha oscillations with amplitude modulation
        alpha = np.sin(2 * np.pi * base_freq * t) * (1 + 0.2 * np.sin(2 * np.pi * 0.3 * t))
        
        # Add some theta (4-7 Hz)
        theta_freq = 6 + 0.2 * np.random.randn()
        theta = 0.3 * np.sin(2 * np.pi * theta_freq * t)
        
        # Add some beta (14-30 Hz)
        beta_freq = 20 + 2 * np.random.randn()
        beta = 0.2 * np.sin(2 * np.pi * beta_freq * t)
        
        # Combine and add noise
        signal_data[ch] = alpha + theta + beta + 0.1 * np.random.randn(n_samples)
    
    return signal_data


def run_cerebral_flow_pipeline(signal_data, sampling_rate=256, n_channels=19):
    """
    Run complete CerebralFlow pipeline.
    
    Parameters:
    -----------
    signal_data : ndarray, shape (n_channels, n_samples)
        Signal data to model
    sampling_rate : float
        Sampling rate of signal data
    n_channels : int
        Number of channels (electrodes)
        
    Returns:
    --------
    refined_model : DynamicOscillatorNetwork
        Final refined model
    metrics : dict
        Performance metrics
    """
    print("Starting CerebralFlow Framework Pipeline")
    
    # Split data into training and validation sets
    n_samples = signal_data.shape[1]
    split_point = int(0.8 * n_samples)
    training_signal = signal_data[:, :split_point]
    validation_signal = signal_data[:, split_point:]
    
    print(f"Data split: Training {training_signal.shape}, Validation {validation_signal.shape}")
    
    # 1. Signal Data Inversion
    print("\nStep 1: Signal Data Inversion")
    inverter = SignalInverter(sampling_rate=sampling_rate)
    inverter.load_data(training_signal)
    phases = inverter.compute_hilbert_phase()
    frequencies = inverter.derive_natural_frequencies()
    connectivity = inverter.assess_connectivity()
    
    print(f"  Extracted phases shape: {phases.shape}")
    print(f"  Estimated frequencies: mean={np.mean(frequencies):.2f} Hz, std={np.std(frequencies):.2f} Hz")
    print(f"  Connectivity matrix: {connectivity.shape}, mean strength={np.mean(connectivity):.4f}")
    
    # 2. Mass Neural Dynamics
    print("\nStep 2: Mass Neural Dynamics")
    mass_dynamics = MassNeuralDynamics(
        n_nodes=n_channels,
        connectivity=connectivity,
        frequencies=frequencies
    )
    _, _, nm_phases, nm_amplitudes = mass_dynamics.simulate(duration=5.0, dt=1.0/sampling_rate)
    
    print(f"  Mass dynamics simulation: phases shape={nm_phases.shape}")
    print(f"  Mass dynamics amplitudes: mean={np.mean(nm_amplitudes):.4f}")
    
    # 3-4. Dynamic Oscillator Network
    print("\nStep 3-4: Dynamic Oscillator Network")
    nm_freqs, nm_coupling = mass_dynamics.get_network_params()
    oscillator_network = DynamicOscillatorNetwork(
        frequencies=nm_freqs,
        adjacency_matrix=nm_coupling,
        base_coupling=1.0,
        inertia=0.1,
        stimulation_strength=0.1,
        noise_level=0.01
    )
    
    print(f"  Oscillator network parameters:")
    print(f"    - Nodes: {oscillator_network.n_nodes}")
    print(f"    - Frequencies: mean={np.mean(oscillator_network.frequencies):.2f} Hz, std={np.std(oscillator_network.frequencies):.2f} Hz")
    print(f"    - Global coupling: {oscillator_network.global_coupling:.4f}")
    
    # 5. Criticality Analysis
    print("\nStep 5: Criticality Analysis")
    criticality_analyzer = CriticalityAnalyzer(oscillator_network)
    mu, meets_threshold = criticality_analyzer.minimum_connectivity_ratio()
    
    print(f"  Connectivity ratio: {mu:.4f}")
    print(f"  Meets critical threshold: {meets_threshold}")
    
    # 6. Generate Signal
    print("\nStep 6: Generate Signal")
    generator = SignalGenerator(oscillator_network)
    sim_duration = training_signal.shape[1] / sampling_rate
    _, network_phases, _ = oscillator_network.simulate(duration=sim_duration, dt=1.0/sampling_rate)
    generated_signal = generator.phase_to_signal(network_phases)
    
    print(f"  Generated signal shape: {generated_signal.shape}")
    
    # 7. Compare Signal
    print("\nStep 7: Compare Signal")
    comparator = SignalComparator(training_signal, generated_signal)
    initial_error = comparator.mean_absolute_error()
    correlations = comparator.pearson_correlation()
    
    print(f"  Initial MAE: {initial_error:.4f}")
    print(f"  Mean correlation: {np.mean(correlations):.4f}")
    
    # 8. Statistical Analysis
    print("\nStep 8: Statistical Analysis")
    analyzer = StatisticalAnalyzer(training_signal, generated_signal)
    real_network, gen_network = analyzer.network_reconstruction()
    network_metrics = analyzer.evaluate_reconstruction(real_network, gen_network)
    
    print(f"  Network reconstruction accuracy: {network_metrics['accuracy']:.4f}")
    print(f"  Network reconstruction precision: {network_metrics['precision']:.4f}")
    
    # 9. Closed-Loop Optimization
    print("\nStep 9: Closed-Loop Optimization")
    optimizer = ClosedLoopOptimizer(
        signal_inverter=inverter,
        mass_dynamics=mass_dynamics,
        oscillator_network=oscillator_network,
        generator=generator,
        comparator=comparator,
        analyzer=analyzer
    )
    
    refined_model, error_history = optimizer.run_optimization(
        training_signal,
        max_iterations=5,
        error_threshold=0.05
    )
    
    print(f"  Optimization complete after {len(error_history)} iterations")
    print(f"  Final error: {error_history[-1]:.4f}")
    
    # Validate refined model
    print("\nValidating refined model on held-out data")
    metrics = optimizer.validate_model(validation_signal)
    
    print("Validation Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Generate final signal with the refined model
    val_duration = validation_signal.shape[1] / sampling_rate
    _, final_phases, _ = refined_model.simulate(
        duration=val_duration, 
        dt=1.0/sampling_rate
    )
    final_signal = generator.phase_to_signal(final_phases)
    
    return refined_model, error_history, final_signal, validation_signal, metrics


def main():
    """
    Main function demonstrating the CerebralFlow framework.
    """
    # Parameters
    sampling_rate = 256  # Hz
    n_channels = 19      # Standard 10-20 system
    duration = 10        # seconds
    
    # Generate synthetic signal data
    print("Generating synthetic signal data")
    signal_data = generate_synthetic_signal(n_channels, duration, sampling_rate)
    
    # Run the framework
    refined_model, error_history, final_signal, validation_signal, metrics = run_cerebral_flow_pipeline(
        signal_data, sampling_rate, n_channels
    )
    
    # Plot results
    print("\nPlotting results")
    plot_full_results(validation_signal, final_signal, refined_model, error_history)
    plt.savefig('results.png')
    # plt.show()
    
    print("\nPipeline completed successfully!")


if __name__ == "__main__":
    main()