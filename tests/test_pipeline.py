"""
Tests for Pipeline
"""

import os
import pytest
from examples.run_pipeline import run_cerebral_flow_pipeline, generate_synthetic_signal
import matplotlib
import matplotlib.pyplot as plt

# Use non-interactive backend
matplotlib.use('Agg')

def test_pipeline_execution():
    # Run with small data for speed
    data = generate_synthetic_signal(n_channels=4, duration=2, sampling_rate=128)
    
    # Mock plt.show/savefig to prevent IO
    original_savefig = plt.savefig
    plt.savefig = lambda *args, **kwargs: None
    
    try:
        model, history, signal, val_signal, metrics = run_cerebral_flow_pipeline(
            data, sampling_rate=128, n_channels=4
        )
        
        assert model is not None
        assert len(history) > 0
        assert signal.shape == val_signal.shape 
    finally:
        plt.savefig = original_savefig
