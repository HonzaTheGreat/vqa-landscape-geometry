# VQA Landscape Geometry

This repository contains Python scripts for simulating and analyzing the discrete Kac-Rice formulation used in VQA landscape studies (Wishart + GOE Hessian model).

The code is script-based on purpose: quick to run, easy to modify, and focused on experiments.

## Repository layout

- `Simulation.py` - main script for Monte Carlo Kac-Rice simulations.
- `Plotting_from_csv.py` - loads saved CSV results and visualizes them.
- `TheoreticalKacRice.py` - simple theoretical baseline computations.
- `tools/` - helper and analysis scripts:
  - `Plotting.py` - shared plotting functions.
  - `Hessian_analysis.py` - Hessian diagnostics and sweeps.
  - `HessDet_noise.py` - sampling/noise behavior checks.
  - `Prefactor.py` - prefactor-only exploration.
  - `FH_Ham_constructor.py` - Fermi-Hubbard Hamiltonian and effective `m` studies.
- `data/` - stored simulation outputs (CSV files).

## Quick start

Use the local virtual environment in this repository:

```bash
source venv/bin/activate
```

Run the main simulation:

```bash
python Simulation.py
```

Plot a saved CSV result:

```bash
python Plotting_from_csv.py
```

Run any helper analysis script:

```bash
python tools/Hessian_analysis.py
python tools/HessDet_noise.py
python tools/Prefactor.py
python tools/FH_Ham_constructor.py
```

## Notes

- Main parameters are configured directly in each script (`m`, `p_values`, energy range, sample size, etc.).
- In `Simulation.py`, use `STORE_RESULTS` and `PLOT_RESULTS` to control CSV saving and plotting.
- Existing CSVs are organized in `data/` subfolders by parameter regime.
