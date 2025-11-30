"""
Criticality Analysis Module
=========================

This module analyzes critical synchronization regime and connectivity thresholds.
It corresponds to the fifth step in the framework.
"""

import numpy as np


class CriticalityAnalyzer:
    """
    Analyze critical synchronization regime and connectivity thresholds.
    
    This class provides tools to analyze the critical synchronization properties
    of the oscillator network, including stability analysis, connectivity thresholds,
    and the identification of synchronized states.
    """
    
    def __init__(self, oscillator_network):
        """
        Initialize analyzer with oscillator network.
        
        Parameters:
        -----------
        oscillator_network : OscillatorNetwork or DynamicOscillatorNetwork
            Model to analyze
        """
        self.model = oscillator_network
        
    def minimum_connectivity_ratio(self):
        """
        Calculate minimum connectivity ratio μ = min_i(degree(i))/(N-1).
        
        Returns:
        --------
        mu : float
            Minimum connectivity ratio
        meets_threshold : bool
            Whether mu >= critical threshold (0.6838)
        """
        adj = self.model.adjacency_matrix
        n = self.model.n_nodes
        
        # Calculate degrees
        degrees = np.sum(adj > 0, axis=1)
        
        # Calculate minimum connectivity ratio
        mu = np.min(degrees) / (n - 1)
        
        # Critical threshold from the paper
        mu_critical = 0.6838
        
        return mu, mu >= mu_critical
    
    def jacobian_stability(self, phases):
        """
        Calculate Jacobian matrix and analyze stability.
        
        Parameters:
        -----------
        phases : ndarray, shape (n_nodes,)
            Phase configuration to analyze
            
        Returns:
        --------
        eigenvalues : ndarray
            Eigenvalues of the Jacobian
        is_stable : bool
            Whether the configuration is stable
        """
        n = self.model.n_nodes
        adj = self.model.adjacency_matrix
        K = self.model.global_coupling
        
        # Initialize Jacobian
        jacobian = np.zeros((n, n))
        
        # Fill Jacobian entries
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Off-diagonal elements
                    jacobian[i, j] = K * adj[i, j] * np.cos(phases[j] - phases[i])
                else:
                    # Diagonal elements
                    jacobian[i, i] = -K * np.sum([adj[i, k] * np.cos(phases[k] - phases[i]) 
                                                 for k in range(n) if k != i])
        
        # Calculate eigenvalues
        eigenvalues = np.linalg.eigvals(jacobian)
        
        # Check stability (all eigenvalues except one should have negative real parts)
        sorted_eigs = np.sort(eigenvalues.real)
        is_stable = np.all(sorted_eigs[:-1] < 0)
        
        return eigenvalues, is_stable
    
    def find_twisted_states(self, n_twists=None):
        """
        Find twisted states configurations.
        
        Parameters:
        -----------
        n_twists : int or list of int, optional
            Number of twists to generate (default: 1 to n//2)
            
        Returns:
        --------
        twisted_states : dict
            Dictionary with twist indices as keys and phase configurations as values
        """
        n = self.model.n_nodes
        
        if n_twists is None:
            n_twists = range(1, n//2 + 1)
        elif isinstance(n_twists, int):
            n_twists = [n_twists]
            
        twisted_states = {}
        
        for p in n_twists:
            # Generate twisted state with p twists
            phases = np.array([2 * np.pi * p * i / n for i in range(n)])
            twisted_states[p] = phases
            
        return twisted_states
    
    def analyze_arnold_tongues(self, frequency_range, coupling_range, 
                               n_freq=20, n_coupling=20, duration=100, dt=0.1):
        """
        Analyze Arnold tongues by varying frequency and coupling.
        
        Parameters:
        -----------
        frequency_range : tuple
            Range of frequency detuning (min, max)
        coupling_range : tuple
            Range of coupling strength (min, max)
        n_freq, n_coupling : int, optional
            Number of points for frequency and coupling
        duration, dt : float, optional
            Simulation parameters
            
        Returns:
        --------
        freq_grid, coupling_grid : ndarray
            Meshgrid of frequency and coupling values
        sync_grid : ndarray
            Synchronization level for each combination
        """
        # Create parameter grids
        freq_values = np.linspace(frequency_range[0], frequency_range[1], n_freq)
        coupling_values = np.linspace(coupling_range[0], coupling_range[1], n_coupling)
        freq_grid, coupling_grid = np.meshgrid(freq_values, coupling_values)
        
        # Initialize synchronization grid
        sync_grid = np.zeros((n_coupling, n_freq))
        
        # Store original parameters
        original_freq = self.model.frequencies.copy()
        original_coupling = self.model.global_coupling
        
        # Loop through parameters
        for i, coupling in enumerate(coupling_values):
            for j, freq_shift in enumerate(freq_values):
                # Update model parameters
                self.model.global_coupling = coupling
                self.model.frequencies = original_freq + freq_shift
                
                # Simulate
                _, _, order_param = self.model.simulate(duration, dt)
                
                # Store mean synchronization from last 20% of simulation
                cutoff = int(0.8 * len(order_param))
                sync_grid[i, j] = np.mean(order_param[cutoff:])
                
        # Reset original parameters
        self.model.frequencies = original_freq
        self.model.global_coupling = original_coupling
        
        return freq_grid, coupling_grid, sync_grid