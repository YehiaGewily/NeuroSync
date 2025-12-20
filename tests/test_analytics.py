"""
Tests for Analytics Modules
"""

import numpy as np
import pytest
from cerebral_flow.analytics.surrogates import SurrogateGenerator, SignificanceTester
from cerebral_flow.analytics.advanced_metrics import phase_lag_index, weighted_phase_lag_index

class TestSurrogates:
    def test_surrogate_generation_shape(self):
        data = np.random.randn(5, 100)
        gen = SurrogateGenerator(data)
        
        surrogates = list(gen.phase_shuffle(n_surrogates=2))
        assert len(surrogates) == 2
        assert surrogates[0].shape == data.shape
        
    def test_phase_shuffle_spectrum(self):
        # Phase shuffling should preserve amplitude spectrum
        t = np.linspace(0, 1, 100)
        data = np.sin(2 * np.pi * 10 * t).reshape(1, -1)
        gen = SurrogateGenerator(data)
        surrogate = next(gen.phase_shuffle(n_surrogates=1))
        
        orig_fft = np.abs(np.fft.rfft(data))
        surr_fft = np.abs(np.fft.rfft(surrogate))
        
        np.testing.assert_allclose(orig_fft, surr_fft, rtol=1e-5, atol=1e-10)

    def test_significance_tester(self):
        obs = 0.8
        dist = [0.1, 0.2, 0.3, 0.2, 0.1]
        p_val, z = SignificanceTester.calculate_p_value(obs, dist, tail='right')
        assert p_val == 0.0 # observed is strictly greater
        assert z > 0

class TestAdvancedMetrics:
    def test_pli_perfect_sync(self):
        # Perfect synchronization (0 lag) should have PLI 0
        t = np.linspace(0, 1, 100)
        phase = 2 * np.pi * 10 * t
        phases = np.vstack([phase, phase]).T # (time, channels)
        
        pli = phase_lag_index(phases)
        assert pli[0, 1] == 0

    def test_pli_lag(self):
        # 90 degree lag should have high PLI
        t = np.linspace(0, 1, 100)
        phase1 = 2 * np.pi * 10 * t
        phase2 = phase1 + np.pi/2
        phases = np.vstack([phase1, phase2]).T
        
        pli = phase_lag_index(phases)
        assert np.isclose(pli[0, 1], 1.0, atol=0.1)
