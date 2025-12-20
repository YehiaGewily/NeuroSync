"""
Example Pipeline Script
======================

This script demonstrates the use of the CerebralFlow framework
by simulating signal data and running the complete pipeline.
"""

import argparse
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

# New Scientific Modules
from cerebral_flow.analytics.surrogates import SurrogateGenerator, SignificanceTester
from cerebral_flow.analytics.advanced_metrics import phase_lag_index, weighted_phase_lag_index


def generate_synthetic_signal(n_channels=19, duration=10, sampling_rate=256):
    """
    Generate synthetic signal data for demonstration.
    """
    n_samples = int(duration * sampling_rate)
    t = np.arange(n_samples) / sampling_rate
    signal_data = np.zeros((n_channels, n_samples))
    
    for ch in range(n_channels):
        # Base frequency in alpha range (8-13 Hz)
        base_freq = 10 + 0.5 * np.random.randn()
        alpha = np.sin(2 * np.pi * base_freq * t) * (1 + 0.2 * np.sin(2 * np.pi * 0.3 * t))
        theta_freq = 6 + 0.2 * np.random.randn()
        theta = 0.3 * np.sin(2 * np.pi * theta_freq * t)
        beta_freq = 20 + 2 * np.random.randn()
        beta = 0.2 * np.sin(2 * np.pi * beta_freq * t)
        signal_data[ch] = alpha + theta + beta + 0.1 * np.random.randn(n_samples)
    
    return signal_data


def run_cerebral_flow_pipeline(signal_data, sampling_rate=256, n_channels=19):
    """
    Run complete CerebralFlow pipeline with scientific validation.
    """
    print("Starting CerebralFlow Framework Pipeline")
    
    # Split data
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
    print(f"  Estimated frequencies: mean={np.mean(frequencies):.2f} Hz")
    
    # --- NEW: Scientific Validation (Surrogates) ---
    print("\nStep 1.5: Statistical Validation (Surrogates)")
    surrogate_gen = SurrogateGenerator(training_signal, sampling_rate)
    n_surrogates = 20
    surr_conn_means = []
    
    print(f"  Generating {n_surrogates} phase-shuffled surrogates...")
    for surrogate in surrogate_gen.phase_shuffle(n_surrogates):
        inverter.load_data(surrogate)
        surr_conn = inverter.assess_connectivity()
        surr_conn_means.append(np.mean(surr_conn))
        
    # Reset inverter to original data
    inverter.load_data(training_signal)
    
    obs_conn_mean = np.mean(connectivity)
    p_val, z_score = SignificanceTester.calculate_p_value(obs_conn_mean, surr_conn_means, tail='right')
    
    print(f"  Observed Mean Connectivity: {obs_conn_mean:.4f}")
    print(f"  Surrogate Mean (N={n_surrogates}): {np.mean(surr_conn_means):.4f}")
    print(f"  Z-score: {z_score:.2f}, p-value: {p_val:.4f}")
    if p_val < 0.05:
        print("  Result is statistically significant (p < 0.05)")
    else:
        print("  Result is NOT statistically significant")
        
    # --- NEW: Advanced Connectivity (PLI) ---
    print("\nStep 1.6: Advanced Connectivity (PLI)")
    # Phases already computed: (n_channels, n_samples) -> transpose for metric
    pli_matrix = phase_lag_index(phases.T)
    print(f"  PLI Matrix Mean: {np.mean(pli_matrix):.4f}")
    
    # 2. Mass Neural Dynamics
    print("\nStep 2: Mass Neural Dynamics")
    mass_dynamics = MassNeuralDynamics(
        n_nodes=n_channels,
        connectivity=connectivity,
        frequencies=frequencies
    )
    _, _, nm_phases, nm_amplitudes = mass_dynamics.simulate(duration=5.0, dt=1.0/sampling_rate)
    
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
    
    # 5. Criticality Analysis
    print("\nStep 5: Criticality Analysis")
    criticality_analyzer = CriticalityAnalyzer(oscillator_network)
    mu, meets_threshold = criticality_analyzer.minimum_connectivity_ratio()
    print(f"  Connectivity ratio: {mu:.4f} (Threshold met: {meets_threshold})")
    
    # 6. Generate Signal
    print("\nStep 6: Generate Signal")
    generator = SignalGenerator(oscillator_network)
    sim_duration = training_signal.shape[1] / sampling_rate
    _, network_phases, _ = oscillator_network.simulate(duration=sim_duration, dt=1.0/sampling_rate)
    generated_signal = generator.phase_to_signal(network_phases)
    
    # 7. Compare Signal
    print("\nStep 7: Compare Signal")
    comparator = SignalComparator(training_signal, generated_signal)
    initial_error = comparator.mean_absolute_error()
    print(f"  Initial MAE: {initial_error:.4f}")
    
    # 8. Statistical Analysis
    print("\nStep 8: Statistical Analysis")
    analyzer = StatisticalAnalyzer(training_signal, generated_signal)
    real_network, gen_network = analyzer.network_reconstruction()
    network_metrics = analyzer.evaluate_reconstruction(real_network, gen_network)
    print(f"  Reconstruction Accuracy: {network_metrics['accuracy']:.4f}")
    
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
    
    print(f"  Optimization complete. Final Error: {error_history[-1]:.4f}")
    
    # Validate
    print("\nValidating refined model...")
    metrics = optimizer.validate_model(validation_signal)
    
    # Generate final signal
    val_duration = validation_signal.shape[1] / sampling_rate
    _, final_phases, _ = refined_model.simulate(duration=val_duration, dt=1.0/sampling_rate)
    final_signal = generator.phase_to_signal(final_phases)
    
    return refined_model, error_history, final_signal, validation_signal, metrics


def main():
    parser = argparse.ArgumentParser(description="Result Pipeline for CerebralFlow")
    parser.add_argument("--duration", type=float, default=10.0, help="Duration of simulation (s)")
    parser.add_argument("--channels", type=int, default=19, help="Number of channels")
    parser.add_argument("--rate", type=int, default=256, help="Sampling rate (Hz)")
    parser.add_argument("--output", type=str, default="results.png", help="Output plot filename")
    
    args = parser.parse_args()
    
    # Generate synthetic signal data
    print("Generating synthetic signal data")
    signal_data = generate_synthetic_signal(args.channels, args.duration, args.rate)
    
    # Run the framework
    refined_model, error_history, final_signal, validation_signal, metrics = run_cerebral_flow_pipeline(
        signal_data, args.rate, args.channels
    )
    
    # Plot results
    print(f"\nPlotting results to {args.output}")
    plot_full_results(validation_signal, final_signal, refined_model, error_history)
    plt.savefig(args.output)
    print("\nPipeline completed successfully!")


if __name__ == "__main__":
    main()
