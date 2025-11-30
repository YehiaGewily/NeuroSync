# NeuroSync: EEG-Kuramoto Digital Twin Framework

![Python](https://img.shields.io/badge/python-3.7%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-Alpha-orange.svg)

**NeuroSync** is a powerful digital twin framework designed to model, analyze, and simulate neural dynamics. By bridging the gap between real EEG data and computational models, it enables researchers to create personalized brain models using the Kuramoto order parameter and neural mass models.

## 🚀 Overview

NeuroSync provides a complete end-to-end pipeline for "digital twinning" of brain dynamics. It takes raw EEG data, extracts the underlying connectivity and natural frequencies, and instantiates a generative model that can reproduce the observed dynamics. This allows for:

*   **Personalized Modeling**: Create models tuned to individual subjects.
*   **In Silico Experimentation**: Test stimulation protocols (like dithered stimulation) on the digital twin before clinical application.
*   **Closed-Loop Refinement**: Automatically tune model parameters to minimize the error between real and simulated brain activity.

## ✨ Key Features

*   **🧠 EEG Data Inversion**: Advanced algorithms to extract phase, frequency, and functional connectivity directly from raw EEG signals.
*   **🕸️ Neural Mass & Kuramoto Modeling**: Hybrid modeling approach combining neural mass dynamics with the phase synchronization properties of the Kuramoto model.
*   **⚡ Dithered Stimulation**: Simulate the effects of external stimulation with time-varying parameters to study entrainment and control.
*   **🔄 Closed-Loop Optimization**: Iterative refinement engine that adjusts model parameters to maximize fidelity to the biological brain.
*   **📊 Comprehensive Analysis**: Built-in tools for critical synchronization analysis, network reconstruction, and statistical validation.

## 🛠️ Installation

You can install the framework directly from the source:

```bash
git clone https://github.com/yourusername/neurosync.git
cd neurosync
pip install -e .
```

### Dependencies
*   Python >= 3.7
*   NumPy, SciPy, Matplotlib
*   NetworkX, Pandas, Scikit-learn
*   MNE-Python

## ⚡ Quick Start

Here is a minimal example to get your first digital twin running:

```python
import numpy as np
from eeg_kuramoto.eeg.data_inversion import EEGDataInversion
from eeg_kuramoto.models.neural_mass import NeuralMassModel
from eeg_kuramoto.models.time_varying import TimeVaryingKuramoto

# 1. Load your EEG data
eeg_data = np.load('subject_data.npy') # shape: (n_channels, n_samples)
sampling_rate = 256

# 2. Invert EEG to get model parameters
inverter = EEGDataInversion(sampling_rate=sampling_rate)
inverter.load_eeg_data(eeg_data)
phases = inverter.extract_phase_hilbert()
frequencies = inverter.estimate_natural_frequency()
connectivity = inverter.estimate_connectivity()

# 3. Initialize the Neural Mass Model
neural_mass = NeuralMassModel(
    n_oscillators=eeg_data.shape[0],
    connectivity=connectivity,
    frequencies=frequencies
)

# 4. Create the Digital Twin (Time-Varying Kuramoto)
nm_freqs, nm_coupling = neural_mass.get_kuramoto_params()
digital_twin = TimeVaryingKuramoto(
    frequencies=nm_freqs,
    adjacency_matrix=nm_coupling,
    stimulation_strength=0.1
)

# 5. Simulate Dynamics
times, sim_phases, order_param = digital_twin.simulate_with_dithering(duration=10.0, dt=1/sampling_rate)
```

For a complete 9-step pipeline demonstration, run the included example:

```bash
python eeg_kuramoto/examples/run_pipeline.py
```

## 📚 Documentation

The framework is organized into modular components:

*   **`eeg_kuramoto.eeg`**: Tools for processing real EEG data and generating synthetic signals.
*   **`eeg_kuramoto.models`**: Core mathematical models (Neural Mass, Standard Kuramoto, Time-Varying Kuramoto).
*   **`eeg_kuramoto.analysis`**: Analytical tools for synchronization, statistics, and closed-loop fitting.
*   **`eeg_kuramoto.utils`**: Helper functions for plotting and metrics.

## 🔬 Scientific Background

NeuroSync is built on the **Kuramoto Model**, a mathematical model used to describe the synchronization of coupled oscillators. By adapting this model with parameters derived from **Neural Mass Models** (which describe the mean activity of neural populations), NeuroSync creates a biologically plausible approximation of large-scale brain dynamics.

## 📄 Citation

If you use NeuroSync in your research, please cite:

```bibtex
@software{neurosync2025,
  title = {NeuroSync: EEG-Kuramoto Digital Twin Framework},
  version = {0.1.0},
  year = {2025},
  author = {The Mind Company}
}
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
