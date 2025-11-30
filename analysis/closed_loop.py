"""
Closed-Loop Refinement Module
============================

This module implements closed-loop refinement to iteratively improve the model.
It corresponds to the ninth and final step in the framework.
"""

import numpy as np
from eeg_kuramoto.models.kuramoto import KuramotoModel
from eeg_kuramoto.models.time_varying import TimeVaryingKuramoto
from eeg_kuramoto.models.neural_mass import NeuralMassModel




class ClosedLoopRefiner:
    """
    Implement closed-loop refinement to iteratively improve the model.
    
    This class orchestrates the iterative refinement of the Kuramoto model
    by integrating all the components of the framework in a closed-loop process
    that adapts model parameters based on the comparison between generated and real EEG.
    """
    
    def __init__(self, eeg_inversion=None, neural_mass=None, kuramoto=None, 
                 generator=None, comparator=None, analyzer=None):
        """
        Initialize refiner with all components.
        
        Parameters:
        -----------
        eeg_inversion : EEGDataInversion, optional
            EEG data inversion component
        neural_mass : NeuralMassModel, optional
            Neural mass model component
        kuramoto : KuramotoModel or TimeVaryingKuramoto, optional
            Kuramoto model component
        generator : EEGGenerator, optional
            EEG generator component
        comparator : EEGComparator, optional
            EEG comparator component
        analyzer : StatisticalAnalyzer, optional
            Statistical analyzer component
        """
        self.eeg_inversion = eeg_inversion
        self.neural_mass = neural_mass
        self.kuramoto = kuramoto
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
        
    def set_components(self, eeg_inversion=None, neural_mass=None, kuramoto=None, 
                       generator=None, comparator=None, analyzer=None):
        """
        Set or update components.
        
        Parameters:
        -----------
        eeg_inversion, neural_mass, kuramoto, generator, comparator, analyzer : optional
            Components to update (only those provided will be updated)
        """
        if eeg_inversion is not None:
            self.eeg_inversion = eeg_inversion
        if neural_mass is not None:
            self.neural_mass = neural_mass
        if kuramoto is not None:
            self.kuramoto = kuramoto
        if generator is not None:
            self.generator = generator
        if comparator is not None:
            self.comparator = comparator
        if analyzer is not None:
            self.analyzer = analyzer
            
    def run_single_iteration(self, real_eeg):
        """
        Run a single iteration of the closed-loop refinement process.
        
        Parameters:
        -----------
        real_eeg : ndarray
            Real EEG data to match
            
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
        components = [self.eeg_inversion, self.neural_mass, self.kuramoto,
                      self.generator, self.comparator, self.analyzer]
        if any(c is None for c in components):
            missing = [name for name, comp in zip(
                ['eeg_inversion', 'neural_mass', 'kuramoto', 'generator', 'comparator', 'analyzer'],
                components
            ) if comp is None]
            raise ValueError(f"Missing components: {', '.join(missing)}")
            
        # Step 1: Invert real EEG to extract parameters
        self.eeg_inversion.load_eeg_data(real_eeg)
        phases = self.eeg_inversion.extract_phase_hilbert()
        frequencies = self.eeg_inversion.estimate_natural_frequency()
        connectivity = self.eeg_inversion.estimate_connectivity()
        
        # Step 2: Update Neural Mass Model with EEG parameters
        if self.iteration == 0:  # Only set initial parameters on first iteration
            self.neural_mass = NeuralMassModel(
                n_oscillators=len(frequencies),
                connectivity=connectivity,
                frequencies=frequencies
            )
        
        # Step 3: Simulate Neural Mass Model
        _, _, nm_phases, _ = self.neural_mass.simulate(duration=5.0, dt=0.01)
        
        # Step 4: Extract Kuramoto parameters from Neural Mass simulation
        nm_freqs, nm_coupling = self.neural_mass.get_kuramoto_params()
        
        # Step 5: Initialize/Update Kuramoto Model
        if self.iteration == 0:  # First iteration
            self.kuramoto = TimeVaryingKuramoto(
                frequencies=nm_freqs,
                adjacency_matrix=nm_coupling,
                base_coupling=1.0,
                stimulation_strength=0.1,
                noise_level=0.01
            )
        else:
            # Update existing model parameters
            self.kuramoto.frequencies = nm_freqs
            self.kuramoto.adjacency_matrix = nm_coupling
        
        # Step 6: Run time-varying Kuramoto with dithered stimulation
        _, kuramoto_phases, _ = self.kuramoto.simulate_with_dithering(duration=5.0, dt=0.01)
        
        # Step 7: Generate EEG from Kuramoto phases
        self.generator.set_kuramoto_model(self.kuramoto)
        generated_eeg = self.generator.phase_to_signal(kuramoto_phases)
        
        # Step 8: Compare generated EEG with real EEG
        self.comparator.set_data(real_eeg, generated_eeg)
        error = self.comparator.mean_absolute_error()
        self.error_history.append(error)
        
        # Step 9: Statistical analysis for model updating
        self.analyzer.set_data(real_eeg, generated_eeg)
        updated_model = self.analyzer.update_model_parameters(self.kuramoto)
        
        # Step 10: Update Kuramoto model for next iteration
        self.kuramoto = updated_model
        
        # Store parameter history
        self.parameter_history['coupling'].append(self.kuramoto.global_coupling)
        self.parameter_history['frequencies'].append(self.kuramoto.frequencies.copy())
        self.parameter_history['connectivity'].append(self.kuramoto.adjacency_matrix.copy())
        
        # Increment iteration counter
        self.iteration += 1
        
        return error
    
    def run_refinement(self, real_eeg, max_iterations=10, error_threshold=0.05):
        """
        Run complete closed-loop refinement process.
        
        Parameters:
        -----------
        real_eeg : ndarray
            Real EEG data to match
        max_iterations : int, optional
            Maximum number of iterations
        error_threshold : float, optional
            Error threshold for early stopping
            
        Returns:
        --------
        final_model : TimeVaryingKuramoto
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
            error = self.run_single_iteration(real_eeg)
            print(f"Iteration {i+1}/{max_iterations}, Error: {error:.4f}")
            
            # Check convergence
            if error < error_threshold:
                print(f"Converged at iteration {i+1} with error {error:.4f}")
                break
                
        return self.kuramoto, self.error_history
    
    def validate_model(self, validation_eeg, duration=10.0, dt=0.01):
        """
        Validate the refined model on new EEG data.
        
        Parameters:
        -----------
        validation_eeg : ndarray
            Validation EEG data
        duration : float, optional
            Simulation duration
        dt : float, optional
            Time step
            
        Returns:
        --------
        metrics : dict
            Validation metrics
        """
        # Run the final model
        _, kuramoto_phases, _ = self.kuramoto.simulate_with_dithering(duration, dt)
        
        # Generate EEG
        self.generator.set_kuramoto_model(self.kuramoto)
        generated_eeg = self.generator.phase_to_signal(kuramoto_phases)
        
        # Compare with validation data
        self.comparator.set_data(validation_eeg, generated_eeg)
        mae = self.comparator.mean_absolute_error()
        correlations = self.comparator.pearson_correlation()
        
        # Statistical analysis
        self.analyzer.set_data(validation_eeg, generated_eeg)
        real_network, gen_network = self.analyzer.network_reconstruction()
        network_metrics = self.analyzer.evaluate_reconstruction(real_network, gen_network)
        
        # Combine metrics
        metrics = {
            'mae': mae,
            'mean_correlation': np.mean(correlations),
            'network_accuracy': network_metrics['accuracy']
        }
        
        return metrics