"""
Closed-Loop Optimization Module
============================

This module implements closed-loop optimization to iteratively improve the model.
It corresponds to the ninth and final step in the framework.
"""

import numpy as np
from cerebral_flow.simulation.kuramoto import OscillatorNetwork
from cerebral_flow.simulation.time_varying import DynamicOscillatorNetwork
from cerebral_flow.simulation.neural_mass import MassNeuralDynamics
from cerebral_flow.signals.data_inversion import SignalInverter
from cerebral_flow.signals.generator import SignalGenerator
from cerebral_flow.signals.comparator import SignalComparator
from cerebral_flow.analytics.statistics import StatisticalAnalyzer


class ClosedLoopOptimizer:
    """
    Implement closed-loop optimization to iteratively improve the model.
    
    This class orchestrates the iterative refinement of the oscillator network
    by integrating all the components of the framework in a closed-loop process
    that adapts model parameters based on the comparison between generated and real signals.
    """
    
    def __init__(self, signal_inverter=None, mass_dynamics=None, oscillator_network=None, 
                 generator=None, comparator=None, analyzer=None):
        """
        Initialize optimizer with all components.
        
        Parameters:
        -----------
        signal_inverter : SignalInverter, optional
            Signal data inversion component
        mass_dynamics : MassNeuralDynamics, optional
            Mass neural dynamics component
        oscillator_network : OscillatorNetwork or DynamicOscillatorNetwork, optional
            Oscillator network component
        generator : SignalGenerator, optional
            Signal generator component
        comparator : SignalComparator, optional
            Signal comparator component
        analyzer : StatisticalAnalyzer, optional
            Statistical analyzer component
        """
        self.signal_inverter = signal_inverter
        self.mass_dynamics = mass_dynamics
        self.oscillator_network = oscillator_network
        self.generator = generator
        self.comparator = comparator
        self.analyzer = analyzer
        
        # For tracking progress
        self.iteration = 0
        self.error_history = []
        self.parameter_history = {
            'coupling': [],
            'frequencies': [],
            'connectivity': []
        }
        
    def set_components(self, signal_inverter=None, mass_dynamics=None, oscillator_network=None, 
                       generator=None, comparator=None, analyzer=None):
        """
        Set or update components.
        
        Parameters:
        -----------
        signal_inverter, mass_dynamics, oscillator_network, generator, comparator, analyzer : optional
            Components to update (only those provided will be updated)
        """
        if signal_inverter is not None:
            self.signal_inverter = signal_inverter
        if mass_dynamics is not None:
            self.mass_dynamics = mass_dynamics
        if oscillator_network is not None:
            self.oscillator_network = oscillator_network
        if generator is not None:
            self.generator = generator
        if comparator is not None:
            self.comparator = comparator
        if analyzer is not None:
            self.analyzer = analyzer
            
    def run_single_iteration(self, real_signal):
        """
        Run a single iteration of the closed-loop optimization process.
        
        Parameters:
        -----------
        real_signal : ndarray
            Real signal data to match
            
        Returns:
        --------
        error : float
            Current error measure
            
        Raises:
        -------
        ValueError
            If any required component is missing
        """
        # Check if all components are initialized
        components = [self.signal_inverter, self.mass_dynamics, self.oscillator_network,
                      self.generator, self.comparator, self.analyzer]
        if any(c is None for c in components):
            missing = [name for name, comp in zip(
                ['signal_inverter', 'mass_dynamics', 'oscillator_network', 'generator', 'comparator', 'analyzer'],
                components
            ) if comp is None]
            raise ValueError(f"Missing components: {', '.join(missing)}")
            
        # Determine simulation parameters
        sampling_rate = self.signal_inverter.sampling_rate
        dt = 1.0 / sampling_rate
        duration = real_signal.shape[1] / sampling_rate
            
        # Step 1: Invert real signal to extract parameters
        self.signal_inverter.load_data(real_signal)
        phases = self.signal_inverter.compute_hilbert_phase()
        frequencies = self.signal_inverter.derive_natural_frequencies()
        connectivity = self.signal_inverter.assess_connectivity()
        
        # Step 2: Update Mass Neural Dynamics with signal parameters
        if self.iteration == 0:  # Only set initial parameters on first iteration
            self.mass_dynamics = MassNeuralDynamics(
                n_nodes=len(frequencies),
                connectivity=connectivity,
                frequencies=frequencies
            )
        
        # Step 3: Simulate Mass Neural Dynamics
        _, _, nm_phases, _ = self.mass_dynamics.simulate(duration=duration, dt=dt)
        
        # Step 4: Extract Oscillator Network parameters from Mass Neural Dynamics simulation
        nm_freqs, nm_coupling = self.mass_dynamics.get_network_params()
        
        # Step 5: Initialize/Update Oscillator Network
        if self.iteration == 0:  # First iteration
            self.oscillator_network = DynamicOscillatorNetwork(
                frequencies=nm_freqs,
                adjacency_matrix=nm_coupling,
                base_coupling=1.0,
                stimulation_strength=0.1,
                noise_level=0.01
            )
        else:
            # Update existing model parameters
            self.oscillator_network.frequencies = nm_freqs
            self.oscillator_network.adjacency_matrix = nm_coupling
        
        # Step 6: Run dynamic oscillator network with dithered stimulation
        _, network_phases, _ = self.oscillator_network.simulate(duration=duration, dt=dt)
        
        # Step 7: Generate signal from network phases
        self.generator.set_oscillator_network(self.oscillator_network)
        generated_signal = self.generator.phase_to_signal(network_phases)
        
        # Step 8: Compare generated signal with real signal
        self.comparator.set_data(real_signal, generated_signal)
        error = self.comparator.mean_absolute_error()
        self.error_history.append(error)
        
        # Step 9: Statistical analysis for model updating
        self.analyzer.set_data(real_signal, generated_signal)
        updated_model = self.analyzer.update_model_parameters(self.oscillator_network)
        
        # Step 10: Update Oscillator Network for next iteration
        self.oscillator_network = updated_model
        
        # Store parameter history
        self.parameter_history['coupling'].append(self.oscillator_network.global_coupling)
        self.parameter_history['frequencies'].append(self.oscillator_network.frequencies.copy())
        self.parameter_history['connectivity'].append(self.oscillator_network.adjacency_matrix.copy())
        
        # Increment iteration counter
        self.iteration += 1
        
        return error
    
    def run_optimization(self, real_signal, max_iterations=10, error_threshold=0.05):
        """
        Run complete closed-loop optimization process.
        
        Parameters:
        -----------
        real_signal : ndarray
            Real signal data to match
        max_iterations : int, optional
            Maximum number of iterations
        error_threshold : float, optional
            Error threshold for early stopping
            
        Returns:
        --------
        final_model : DynamicOscillatorNetwork
            Final refined model
        error_history : list
            Error at each iteration
        """
        self.iteration = 0
        self.error_history = []
        self.parameter_history = {
            'coupling': [],
            'frequencies': [],
            'connectivity': []
        }
        
        for i in range(max_iterations):
            error = self.run_single_iteration(real_signal)
            print(f"Iteration {i+1}/{max_iterations}, Error: {error:.4f}")
            
            # Check convergence
            if error < error_threshold:
                print(f"Converged at iteration {i+1} with error {error:.4f}")
                break
                
        return self.oscillator_network, self.error_history
    
    def validate_model(self, validation_signal, duration=None, dt=None):
        """
        Validate the refined model on new signal data.
        
        Parameters:
        -----------
        validation_signal : ndarray
            Validation signal data
        duration : float, optional
            Simulation duration. If None, calculated from signal.
        dt : float, optional
            Time step. If None, calculated from signal inverter sampling rate.
            
        Returns:
        --------
        metrics : dict
            Validation metrics
        """
        # Determine simulation parameters if not provided
        if duration is None or dt is None:
            sampling_rate = self.signal_inverter.sampling_rate
            if dt is None:
                dt = 1.0 / sampling_rate
            if duration is None:
                duration = validation_signal.shape[1] / sampling_rate

        # Run the final model
        _, network_phases, _ = self.oscillator_network.simulate(duration, dt)
        
        # Generate signal
        self.generator.set_oscillator_network(self.oscillator_network)
        generated_signal = self.generator.phase_to_signal(network_phases)
        
        # Compare with validation data
        self.comparator.set_data(validation_signal, generated_signal)
        mae = self.comparator.mean_absolute_error()
        correlations = self.comparator.pearson_correlation()
        
        # Statistical analysis
        self.analyzer.set_data(validation_signal, generated_signal)
        real_network, gen_network = self.analyzer.network_reconstruction()
        network_metrics = self.analyzer.evaluate_reconstruction(real_network, gen_network)
        
        # Combine metrics
        metrics = {
            'mae': mae,
            'mean_correlation': np.mean(correlations),
            'network_accuracy': network_metrics['accuracy']
        }
        
        return metrics