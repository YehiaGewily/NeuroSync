"""
Analysis modules for the EEG-Kuramoto Neural Dynamics Framework.
"""

from .critical_sync import CriticalSynchronizationAnalyzer
from .statistics import StatisticalAnalyzer
from .closed_loop import ClosedLoopRefiner

__all__ = ['CriticalSynchronizationAnalyzer', 'StatisticalAnalyzer', 'ClosedLoopRefiner']