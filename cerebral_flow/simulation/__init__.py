"""
Simulation Package
=================

This package contains modules for simulating neural dynamics, including
oscillator networks and mass neural dynamics.
"""

from .neural_mass import MassNeuralDynamics
from .kuramoto import OscillatorNetwork
from .time_varying import DynamicOscillatorNetwork

__all__ = ['MassNeuralDynamics', 'OscillatorNetwork', 'DynamicOscillatorNetwork']