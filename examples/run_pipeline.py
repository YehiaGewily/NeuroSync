"""
Example Pipeline Script
======================

This script demonstrates the use of the EEG-Kuramoto Neural Dynamics framework
by simulating EEG data and running the complete pipeline.
"""

import numpy as np
import matplotlib.pyplot as plt

# Import framework components
from eeg_kuramoto.eeg.data_inversion import EEGDataInversion
from eeg_kuramoto.models.neural_mass import NeuralMassModel
from eeg_kuramoto.models.time_varying import TimeVaryingKuramoto
from eeg_kuramoto.analysis.critical_sync import CriticalSynchronizationAnalyzer
from eeg_kuramoto.eeg.generator import EEGGenerator
from eeg_kuramoto.eeg.comparator import EEGComparator
from eeg_kuramoto.analysis.statistics import StatisticalAnalyzer
from eeg_kuramoto.analysis.closed_loop import ClosedLoopRefiner
from eeg_kuramoto.utils.plotting import plot_full_results
from eeg_kuramoto.utils.metrics import compare_networks


def generate_synthetic_eeg(n_channels=19, duration=10, sampling_rate=256):
    """
    Generate synthetic EEG data for demonstration.
    
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
    eeg_data : ndarray, shape (n_channels, n_samples)
        Synthetic EEG data
    """
    n_samples = int(duration * sampling_rate)
    
    # Create time vector
    t = np.arange(n_samples) / sampling_rate
    
    # Initialize EEG data
    eeg_data = np.zeros((n_channels, n_samples))
    
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
        eeg_data[ch] = alpha + theta + beta + 0.1 * np.random.randn(n_samples)
    
    return eeg_data


def run_eeg_kuramoto_pipeline(eeg_data, sampling_rate=256, n_channels=19):
    """
    Run complete EEG-Kuramoto Neural Dynamics pipeline.
    
    Parameters:
    -----------
    eeg_data : ndarray, shape (n_channels, n_samples)
        EEG data to model
    sampling_rate : float
        Sampling rate of EEG data
    n_channels : int
        Number of channels (electrodes)
        
    Returns:
    --------
    refined_model : TimeVaryingKuramoto
        Final refined model
    metrics : dict
        Performance metrics
    """
    print("Starting EEG-Kuramoto Neural Dynamics Framework Pipeline")
    
    # Split data into training and validation sets
    n_samples = eeg_data.shape[1]
    split_point = int(0.8 * n_samples)
    training_eeg = eeg_data[:, :split_point]
    validation_eeg = eeg_data[:, split_point:]
    
    print(f"Data split: Training {training_eeg.shape}, Validation {validation_eeg.shape}")
    
    # 1. EEG Data Inversion
    print("\nStep 1: EEG Data Inversion")
    eeg_inverter = EEGDataInversion(sampling_rate=sampling_rate)
    eeg_inverter.load_eeg_data(training_eeg)
    phases = eeg_inverter.extract_phase_hilbert()
    frequencies = eeg_inverter.estimate_natural_frequency()
    connectivity = eeg_inverter.estimate_connectivity()
    
    print(f"  Extracted phases shape: {phases.shape}")
    print(f"  Estimated frequencies: mean={np.mean(frequencies):.2f} Hz, std={np.std(frequencies):.2f} Hz")
    print(f"  Connectivity matrix: {connectivity.shape}, mean strength={np.mean(connectivity):.4f}")
    
    # 2. Neural Mass Model
    print("\nStep 2: Neural Mass Model")
    neural_mass = NeuralMassModel(
        n_oscillators=n_channels,
        connectivity=connectivity,
        frequencies=frequencies
    )
    _, _, nm_phases, nm_amplitudes = neural_mass.simulate(duration=5.0, dt=1.0/sampling_rate)
    
    print(f"  Neural mass simulation: phases shape={nm_phases.shape}")
    print(f"  Neural mass amplitudes: mean={np.mean(nm_amplitudes):.4f}")
    
    # 3-4. Kuramoto Model with Time-Varying Parameters
    print("\nStep 3-4: Kuramoto Model with Time-Varying Parameters")
    nm_freqs, nm_coupling = neural_mass.get_kuramoto_params()
    kuramoto = TimeVaryingKuramoto(
        frequencies=nm_freqs,
        adjacency_matrix=nm_coupling,
        base_coupling=1.0,
        inertia=0.1,
        stimulation_strength=0.1,
        noise_level=0.01
    )
    
    print(f"  Kuramoto model parameters:")
    print(f"    - Oscillators: {kuramoto.n_oscillators}")
    print(f"    - Frequencies: mean={np.mean(kuramoto.frequencies):.2f} Hz, std={np.std(kuramoto.frequencies):.2f} Hz")
    print(f"    - Global coupling: {kuramoto.global_coupling:.4f}")
    
    # 5. Critical Synchronization Analysis
    print("\nStep 5: Critical Synchronization Analysis")
    sync_analyzer = CriticalSynchronizationAnalyzer(kuramoto)
    mu, meets_threshold = sync_analyzer.minimum_connectivity_ratio()
    
    print(f"  Connectivity ratio: {mu:.4f}")
    print(f"  Meets critical threshold: {meets_threshold}")
    
    # 6. Generate EEG
    print("\nStep 6: Generate EEG")
    generator = EEGGenerator(kuramoto)
    _, kuramoto_phases, _ = kuramoto.simulate_with_dithering(duration=5.0, dt=1.0/sampling_rate)
    generated_eeg = generator.phase_to_signal(kuramoto_phases)
    
    print(f"  Generated EEG shape: {generated_eeg.shape}")
    
    # 7. Compare EEG
    print("\nStep 7: Compare EEG")
    comparator = EEGComparator(training_eeg, generated_eeg)
    initial_error = comparator.mean_absolute_error()
    correlations = comparator.pearson_correlation()
    
    print(f"  Initial MAE: {initial_error:.4f}")
    print(f"  Mean correlation: {np.mean(correlations):.4f}")
    
    # 8. Statistical Analysis
    print("\nStep 8: Statistical Analysis")
    analyzer = StatisticalAnalyzer(training_eeg, generated_eeg)
    real_network, gen_network = analyzer.network_reconstruction()
    network_metrics = analyzer.evaluate_reconstruction(real_network, gen_network)
    
    print(f"  Network reconstruction accuracy: {network_metrics['accuracy']:.4f}")
    print(f"  Network reconstruction precision: {network_metrics['precision']:.4f}")
    
    # 9. Closed-Loop Refinement
    print("\nStep 9: Closed-Loop Refinement")
    refiner = ClosedLoopRefiner(
        eeg_inversion=eeg_inverter,
        neural_mass=neural_mass,
        kuramoto=kuramoto,
        generator=generator,
        comparator=comparator,
        analyzer=analyzer
    )
    
    refined_model, error_history = refiner.run_refinement(
        training_eeg,
        max_iterations=5,
        error_threshold=0.05
    )
    
    print(f"  Refinement complete after {len(error_history)} iterations")
    print(f"  Final error: {error_history[-1]:.4f}")
    
    # Validate refined model
    print("\nValidating refined model on held-out data")
    metrics = refiner.validate_model(validation_eeg)
    
    print("Validation Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Generate final EEG with the refined model
    _, final_phases, _ = refined_model.simulate_with_dithering(
        duration=len(validation_eeg[0]) / sampling_rate, 
        dt=1.0/sampling_rate
    )
    final_eeg = generator.phase_to_signal(final_phases)
    
    return refined_model, error_history, final_eeg, validation_eeg, metrics


def main():
    """
    Main function demonstrating the EEG-Kuramoto Neural Dynamics framework.
    """
    # Parameters
    sampling_rate = 256  # Hz
    n_channels = 19      # Standard 10-20 system
    duration = 10        # seconds
    
    # Generate synthetic EEG data
    print("Generating synthetic EEG data")
    eeg_data = generate_synthetic_eeg(n_channels, duration, sampling_rate)
    
    # Run the framework
    refined_model, error_history, final_eeg, validation_eeg, metrics = run_eeg_kuramoto_pipeline(
        eeg_data, sampling_rate, n_channels
    )
    
    # Plot results
    print("\nPlotting results")
    plot_full_results(validation_eeg, final_eeg, refined_model, error_history)
    plt.show()
    
    print("\nPipeline completed successfully!")


if __name__ == "__main__":
    main()