# CerebralFlow: Neural Dynamics Simulation Framework

![Python](https://img.shields.io/badge/python-3.7%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-Alpha-orange.svg)

**CerebralFlow** is a comprehensive framework for modeling and simulating large-scale neural dynamics. It provides tools to reconstruct functional networks from signal data and simulate brain activity using coupled oscillator models.

##  Overview

CerebralFlow enables the creation of "digital twins" of brain dynamics. It processes physiological signals to extract connectivity and intrinsic frequencies, then instantiates generative models to reproduce observed states.

*   **Personalized Modeling**: Data-driven model construction.
*   **Simulation Engine**: Efficient simulation of neural mass and oscillator networks.
*   **Closed-Loop Control**: Optimization tools for model fitting and control parameter tuning.

##  Key Features

*   **Signal Analytics**: Hilbert transform-based phase extraction and frequency estimation.
*   **Simulation Models**:
    *   **MassNeuralDynamics**: Mean-field approximations of neural populations.
    *   **DynamicOscillatorNetwork**: Time-varying Kuramoto models for synchronization studies.
*   **Connectivity Analysis**: Functional connectivity estimation from time-series data.
*   **Dithered Control**: Simulation of external stimulation effects.

## Installation

```bash
git clone https://github.com/cerebralflow/cerebralflow.git
cd cerebralflow
pip install -e .
```

## Quick Start

```python
import numpy as np
from cerebral_flow.signals.data_inversion import SignalInverter
from cerebral_flow.simulation.neural_mass import MassNeuralDynamics
from cerebral_flow.simulation.time_varying import DynamicOscillatorNetwork

# 1. Load data
data = np.load('subject_data.npy') # shape: (n_channels, n_samples)
fs = 256

# 2. Invert signals to model parameters
inverter = SignalInverter(sampling_rate=fs)
inverter.load_data(data)
phases = inverter.compute_hilbert_phase()
freqs = inverter.derive_natural_frequencies()
conn = inverter.assess_connectivity()

# 3. Initialize Dynamics
model = MassNeuralDynamics(
    n_nodes=data.shape[0],
    connectivity=conn,
    frequencies=freqs
)

# 4. Create Simulation
sim_freqs, sim_coupling = model.get_network_params()
simulation = DynamicOscillatorNetwork(
    frequencies=sim_freqs,
    adjacency_matrix=sim_coupling,
    coupling_strength=0.1
)

# 5. Run Simulation
times, phases, order = simulation.simulate(duration=10.0, dt=1/fs)
```

## Documentation

*   **`cerebral_flow.signals`**: Signal processing and inversion tools.
*   **`cerebral_flow.simulation`**: Core simulation models.
*   **`cerebral_flow.analytics`**: Analysis and metrics.
*   **`cerebral_flow.common`**: Utilities and helpers.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
