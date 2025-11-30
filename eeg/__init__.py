"""
EEG processing modules for the EEG-Kuramoto Neural Dynamics Framework.
"""

from .data_inversion import EEGDataInversion
from .generator import EEGGenerator
from .comparator import EEGComparator

__all__ = ['EEGDataInversion', 'EEGGenerator', 'EEGComparator']