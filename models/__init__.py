"""
Model implementations for the EEG-Kuramoto Neural Dynamics Framework.
"""

from .neural_mass import NeuralMassModel
from .kuramoto import KuramotoModel
from .time_varying import TimeVaryingKuramoto

__all__ = ['NeuralMassModel', 'KuramotoModel', 'TimeVaryingKuramoto']