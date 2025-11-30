"""
Dynamic Oscillator Network Module
=================================

This module implements the oscillator network with time-varying parameters and dithered stimulation.
It corresponds to the fourth step in the framework.
"""

import numpy as np
from scipy.integrate import solve_ivp
from .kuramoto import OscillatorNetwork


class DynamicOscillatorNetwork(OscillatorNetwork):
    """
    Oscillator network with time-varying parameters and dithered stimulation.
    
    This class extends the basic oscillator network to include time-varying coupling,
    external stimulation, noise (dithering), and inertial effects.
    """
    
    def __init__(self, frequencies, adjacency_matrix, base_coupling=1.0, 
                 inertia=0.0, stimulation_strength=0.0, noise_level=0.0):
        """
        Initialize dynamic oscillator network.
        
        Parameters:
        -----------
        frequencies : ndarray, shape (n_nodes,)
            Natural frequencies for oscillators
        adjacency_matrix : ndarray, shape (n_nodes, n_nodes)
            Connectivity matrix between oscillators
        base_coupling : float, optional
            Base coupling strength
        inertia : float, optional
            Inertial term coefficient
        stimulation_strength : float, optional
            Stimulation intensity
        noise_level : float, optional
            Noise intensity for dithering
        """
        super().__init__(frequencies, adjacency_matrix, base_coupling)
        
        self.inertia = inertia
        self.stimulation_strength = stimulation_strength
        self.noise_level = noise_level
        
        # For inertial model
        self.velocities = np.zeros(self.n_nodes)
        
        # Sensitivity to external input (can be heterogeneous)
        self.sensitivity = np.ones(self.n_nodes)
        
        # For storing time-varying parameters
        self.coupling_history = None
        self.input_history = None
        
    def time_varying_coupling(self, t):
        """
        Time-varying coupling function (e.g., periodic).
        
        Parameters:
        -----------
        t : float
            Time point
            
        Returns:
        --------
        coupling : float
            Coupling strength at time t
        """
        return self.global_coupling * (1.0 + 0.2 * np.sin(0.1 * t))
    
    def external_input(self, t):
        """
        External input function (e.g., stimulus).
        
        Parameters:
        -----------
        t : float
            Time point
            
        Returns:
        --------
        input : float
            External input at time t
        """
        return self.stimulation_strength * np.sin(2.0 * t)
    
    def dithered_phase_evolution(self, t, state):
        """
        Phase evolution with time-varying parameters, external input, and noise.
        
        Parameters:
        -----------
        t : float
            Time point
        state : ndarray, shape (2*n_nodes,)
            Current phases and velocities (for inertial model)
            
        Returns:
        --------
        dstate_dt : ndarray, shape (2*n_nodes,)
            Derivatives of phases and velocities
        """
        n = self.n_nodes
        phases = state[:n]
        velocities = state[n:] if self.inertia > 0 else np.zeros(n)
        
        dphases_dt = np.zeros(n)
        dvelocities_dt = np.zeros(n)
        
        # Get time-varying coupling
        K = self.time_varying_coupling(t)
        
        # External input
        input_signal = self.external_input(t)
        
        # Generate noise for dithering
        noise = np.random.normal(0, self.noise_level, n) if self.noise_level > 0 else np.zeros(n)
        
        for i in range(n):
            # For inertial model
            if self.inertia > 0:
                dphases_dt[i] = velocities[i]
                
                # Inertial term
                acceleration = -velocities[i]  # Damping
                
                # Natural frequency and coupling terms
                acceleration += self.frequencies[i]
                
                for j in range(n):
                    if self.adjacency_matrix[i, j] > 0:
                        acceleration += K * self.adjacency_matrix[i, j] * np.sin(phases[j] - phases[i])
                
                # External input and noise
                acceleration += self.sensitivity[i] * input_signal + noise[i]
                
                dvelocities_dt[i] = acceleration / self.inertia
                
            else:
                # Standard oscillator network with time-varying parameters
                dphases_dt[i] = self.frequencies[i]
                
                # Coupling term
                for j in range(n):
                    if self.adjacency_matrix[i, j] > 0:
                        dphases_dt[i] += K * self.adjacency_matrix[i, j] * np.sin(phases[j] - phases[i])
                
                # External input and noise
                dphases_dt[i] += self.sensitivity[i] * input_signal + noise[i]
        
        if self.inertia > 0:
            return np.concatenate((dphases_dt, dvelocities_dt))
        else:
            return dphases_dt
    
    def simulate(self, duration, dt, initial_phases=None):
        """
        Simulate dynamic oscillator network.
        
        Parameters:
        -----------
        duration : float
            Simulation duration in time units
        dt : float
            Time step for simulation
        initial_phases : ndarray, shape (n_nodes,), optional
            Initial phases
            
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
            
        # For inertial model, include velocities in state
        if self.inertia > 0:
            initial_state = np.concatenate((initial_phases, np.zeros(self.n_nodes)))
        else:
            initial_state = initial_phases
            
        # Set up arrays for results
        self.phases = np.zeros((n_steps, self.n_nodes))
        if self.inertia > 0:
            self.phases[0, :] = initial_state[:self.n_nodes]
        else:
            self.phases[0, :] = initial_state
            
        self.order_parameter = np.zeros(n_steps)
        self.order_parameter[0] = self.calculate_order_parameter(self.phases[0, :])
        
        # For storing time-varying coupling and input
        self.coupling_history = np.zeros(n_steps)
        self.input_history = np.zeros(n_steps)
        
        # Simulate using solve_ivp
        solution = solve_ivp(
            self.dithered_phase_evolution,
            [0, duration],
            initial_state,
            t_eval=self.times,
            method='RK45'
        )
        
        # Extract results
        if self.inertia > 0:
            self.phases = solution.y[:self.n_nodes, :].T
            self.velocities = solution.y[self.n_nodes:, :].T
        else:
            self.phases = solution.y.T
            
        # Calculate order parameter
        self.order_parameter = self.calculate_order_parameter(self.phases)
        
        # Store time-varying parameters
        for i, t in enumerate(self.times):
            self.coupling_history[i] = self.time_varying_coupling(t)
            self.input_history[i] = self.external_input(t)
            
        return self.times, self.phases, self.order_parameter