"""
Oscillator Network Module
====================

This module implements the oscillator network model using parameters derived from mass neural dynamics.
It corresponds to the third step in the framework.
"""

import numpy as np
from scipy.integrate import solve_ivp


class OscillatorNetwork:
    """
    Implementation of the oscillator network model using parameters derived from mass neural dynamics.
    
    The oscillator network describes a system of coupled oscillators and is used to
    model synchronization phenomena in neural systems.
    """
    
    def __init__(self, frequencies, adjacency_matrix, global_coupling=1.0):
        """
        Initialize oscillator network.
        
        Parameters:
        -----------
        frequencies : ndarray, shape (n_nodes,)
            Natural frequencies for oscillators
        adjacency_matrix : ndarray, shape (n_nodes, n_nodes)
            Connectivity matrix between oscillators
        global_coupling : float, optional
            Global coupling strength
        """
        self.n_nodes = len(frequencies)
        self.frequencies = frequencies
        self.adjacency_matrix = adjacency_matrix
        self.global_coupling = global_coupling
        
        # Storage for simulation results
        self.times = None
        self.phases = None
        self.order_parameter = None
        
    def phase_evolution(self, t, phases, K=None):
        """
        Basic oscillator network differential equation.
        
        Parameters:
        -----------
        t : float
            Time point
        phases : ndarray, shape (n_nodes,)
            Current phases of oscillators
        K : float, optional
            Current coupling strength (for time-varying coupling)
            
        Returns:
        --------
        dphases_dt : ndarray, shape (n_nodes,)
            Phase derivatives
        """
        if K is None:
            K = self.global_coupling
            
        dphases_dt = np.zeros(self.n_nodes)
        
        for i in range(self.n_nodes):
            # Natural frequency term
            dphases_dt[i] = self.frequencies[i]
            
            # Coupling term
            for j in range(self.n_nodes):
                if self.adjacency_matrix[i, j] > 0:
                    dphases_dt[i] += K * self.adjacency_matrix[i, j] * np.sin(phases[j] - phases[i])
                    
        return dphases_dt
    
    def calculate_order_parameter(self, phases):
        """
        Calculate order parameter (measure of synchronization).
        
        Parameters:
        -----------
        phases : ndarray, shape (n_nodes,) or (n_time_points, n_nodes)
            Phases of oscillators
            
        Returns:
        --------
        r : float or ndarray
            Order parameter (0 ≤ r ≤ 1), where 1 indicates perfect synchronization
        """
        if phases.ndim == 1:
            # Single time point
            z = np.mean(np.exp(1j * phases))
            return np.abs(z)
        else:
            # Multiple time points
            z = np.mean(np.exp(1j * phases), axis=1)
            return np.abs(z)
    
    def simulate(self, duration, dt, initial_phases=None):
        """
        Simulate oscillator network.
        
        Parameters:
        -----------
        duration : float
            Simulation duration in time units
        dt : float
            Time step for simulation
        initial_phases : ndarray, shape (n_nodes,), optional
            Initial phases (default: random from uniform distribution [0, 2π])
            
        Returns:
        --------
        times : ndarray
            Time points
        phases : ndarray, shape (n_time_points, n_nodes)
            Phase evolution
        order_parameter : ndarray, shape (n_time_points,)
            Order parameter evolution
        """
        # Create time points
        self.times = np.arange(0, duration, dt)
        n_steps = len(self.times)
        
        # Initialize phases if not provided
        if initial_phases is None:
            initial_phases = np.random.uniform(0, 2*np.pi, self.n_nodes)
            
        # Simulate using solve_ivp for better accuracy
        solution = solve_ivp(
            self.phase_evolution,
            [0, duration],
            initial_phases,
            t_eval=self.times,
            method='RK45'
        )
        
        self.phases = solution.y.T
        self.order_parameter = self.calculate_order_parameter(self.phases)
            
        return self.times, self.phases, self.order_parameter