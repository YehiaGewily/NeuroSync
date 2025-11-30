"""
Mass Neural Dynamics Module
========================

This module implements the mass neural dynamics model bridging signal data to oscillator networks.
It corresponds to the second step in the framework.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy import signal


class MassNeuralDynamics:
    """
    Mass neural dynamics model that bridges signal data to oscillator networks.
    
    This class implements a neural mass model based on the Jansen-Rit model,
    which simulates the dynamics of neural populations to generate oscillatory
    activity resembling physiological signals.
    """
    
    def __init__(self, n_nodes, connectivity=None, frequencies=None):
        """
        Initialize mass neural dynamics model.
        
        Parameters:
        -----------
        n_nodes : int
            Number of neural masses (nodes)
        connectivity : ndarray, shape (n_nodes, n_nodes), optional
            Connectivity matrix between neural masses
        frequencies : ndarray, shape (n_nodes,), optional
            Intrinsic frequencies of neural masses
        """
        self.n_nodes = n_nodes
        
        # Initialize parameters
        if connectivity is None:
            self.connectivity = np.ones((n_nodes, n_nodes))
            np.fill_diagonal(self.connectivity, 0)
        else:
            self.connectivity = connectivity
            
        if frequencies is None:
            self.frequencies = np.random.normal(10, 1, n_nodes)  # Mean 10Hz, SD 1Hz
        else:
            self.frequencies = frequencies
            
        # Neural mass parameters (Jansen-Rit model)
        self.excitatory_time_constant = 10.0  # ms
        self.inhibitory_time_constant = 20.0  # ms
        self.excitatory_amplitude = 3.25
        self.inhibitory_amplitude = 22.0
        self.sigmoid_slope = 0.56
        self.sigmoid_threshold = 6.0
        
        # State variables
        self.x = np.zeros((n_nodes, 4))  # [y0, y1, y2, y3] for each node
        
        # Output variables
        self.phases = None
        self.amplitudes = None
        
    def sigmoid(self, v):
        """
        Sigmoid activation function for neural masses.
        
        Parameters:
        -----------
        v : float or ndarray
            Input value(s)
            
        Returns:
        --------
        output : float or ndarray
            Sigmoid activation output
        """
        return 2 * self.excitatory_amplitude / (1 + np.exp(self.sigmoid_slope * (self.sigmoid_threshold - v)))
    
    def derivatives(self, t, x):
        """
        Neural mass model differential equations.
        
        Parameters:
        -----------
        t : float
            Time point
        x : ndarray, shape (4*n_nodes,)
            State variables flattened
            
        Returns:
        --------
        dx_dt : ndarray, shape (4*n_nodes,)
            Derivatives of state variables
        """
        # Reshape x to (n_nodes, 4)
        x_reshaped = x.reshape(self.n_nodes, 4)
        dx_dt = np.zeros_like(x_reshaped)
        
        for i in range(self.n_nodes):
            # Extract state variables for this neural mass
            y0, y1, y2, y3 = x_reshaped[i]
            
            # Calculate input from other neural masses
            coupling_input = 0
            for j in range(self.n_nodes):
                if self.connectivity[i, j] > 0:
                    coupling_input += self.connectivity[i, j] * x_reshaped[j, 0]  # y0 is the output
            
            # Equations for excitatory population
            dx_dt[i, 0] = y1
            dx_dt[i, 1] = (self.excitatory_amplitude * self.sigmoid(y2 - y3) - 2 * y1 - y0) / self.excitatory_time_constant
            
            # Equations for inhibitory population
            dx_dt[i, 2] = y3
            dx_dt[i, 3] = (self.inhibitory_amplitude * self.sigmoid(y0) - 2 * y3 - y2) / self.inhibitory_time_constant
            
            # Add coupling and frequency-specific inputs
            dx_dt[i, 1] += self.frequencies[i] * 0.1 + coupling_input * 0.01
        
        return dx_dt.flatten()
    
    def simulate(self, duration, dt, initial_conditions=None):
        """
        Simulate neural mass model.
        
        Parameters:
        -----------
        duration : float
            Simulation duration in seconds
        dt : float
            Time step for simulation
        initial_conditions : ndarray, optional
            Initial conditions for state variables
            
        Returns:
        --------
        times : ndarray
            Time points
        states : ndarray, shape (n_time_points, n_nodes, 4)
            State variable evolution
        phases : ndarray, shape (n_time_points, n_nodes)
            Extracted phases
        amplitudes : ndarray, shape (n_time_points, n_nodes)
            Extracted amplitudes
        """
        # Create time points
        times = np.arange(0, duration, dt)
        n_steps = len(times)
        
        # Initialize state variables if not provided
        if initial_conditions is None:
            initial_conditions = np.random.uniform(-0.1, 0.1, (self.n_nodes, 4)).flatten()
            
        # Simulate using solve_ivp
        solution = solve_ivp(
            self.derivatives,
            [0, duration],
            initial_conditions,
            t_eval=times,
            method='RK45'
        )
        
        # Reshape solution
        states = solution.y.T.reshape(n_steps, self.n_nodes, 4)
        
        # Extract output variables
        output = states[:, :, 0]  # y0 is the main output
        
        # Compute phases and amplitudes using Hilbert transform
        self.phases = np.zeros((n_steps, self.n_nodes))
        self.amplitudes = np.zeros((n_steps, self.n_nodes))
        
        for i in range(self.n_nodes):
            analytic_signal = signal.hilbert(output[:, i])
            self.phases[:, i] = np.angle(analytic_signal)
            self.amplitudes[:, i] = np.abs(analytic_signal)
            
        return times, states, self.phases, self.amplitudes
    
    def get_network_params(self):
        """
        Extract parameters for oscillator network from neural mass simulation.
        
        Returns:
        --------
        freq : ndarray, shape (n_nodes,)
            Natural frequencies for oscillator network
        coupling : ndarray, shape (n_nodes, n_nodes)
            Coupling matrix for oscillator network
            
        Raises:
        -------
        ValueError
            If no simulation data is available
        """
        if self.phases is None:
            raise ValueError("No simulation data available. Run simulate() first.")
            
        # Estimate frequencies from phases
        n_steps = self.phases.shape[0]
        freq = np.zeros(self.n_nodes)
        
        for i in range(self.n_nodes):
            # Calculate mean frequency
            unwrapped = np.unwrap(self.phases[:, i])
            freq[i] = (unwrapped[-1] - unwrapped[0]) / (n_steps - 1) * 1000  # Convert to Hz
            
        # Use connectivity matrix as initial coupling
        coupling = self.connectivity.copy()
        
        return freq, coupling