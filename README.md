# QLBM Simulation with ZNE Mitigation

This repository contains a quantum lattice Boltzmann method (QLBM) simulation implementation. It uses the [`qlbm`](https://github.com/QCFD-Lab/qlbm) framework [1] with a Qiskit Aer quantum simulator backend, and contains built-in error mitigation via zero-noise extrapolation (ZNE).

## Project Structure

* **`main.py`**: Configures simulation parameters (lattice size, shots, noise models) and runs simulations.
* **`qlbm_tools.py`**: Utilities:
    * **Backend Spy**: Intercepts Qiskit jobs to perform ZNE gate folding.
    * **Visualization**: Generates `.vti` files and visualisations (gif and png).
    * **Analysis**: Calculates RMSE against ideal simulations and plots convergence graphs.
* **`requirements.txt`**: List of Python dependencies.

## Installation

**Prerequisites:** Python 3.12

Due to specific version requirements for the `qlbm` framework and `mitiq`, please follow these steps exactly.

1.  **Clone and Setup Environment**
    ```bash
    git clone [https://github.com/QCFD-Lab/qlbm.git](https://github.com/QCFD-Lab/qlbm.git)
    cd qlbm
    py -3.12 -m venv .venv
    .\.venv\Scripts\activate
    pip install --upgrade pip
    ```

2.  **Install Dependencies**
    ```bash
    pip install -e .[cpu]
    pip install mitiq --no-deps
    pip install -r requirements.txt
    ```

## Usage

1.  **Configure Parameters**:
    Open `main.py` and modify the `PARAMS` dictionary to set your simulation geometry, shot counts, and noise levels.

    ```python
    PARAMS = {
        "d": 4,                        # lattice dimension (d x d)
        "shots": [4096, 8192],         # shot counts to sweep over
        "steps": 3,                    # simulation steps (not including initial step)
        "noise_model": "depolarizing", # "depolarizing" or "fakefez"
        "zne_scales": [1.0, 3.0, 5.0], # ZNE scaling factors
        "p_err_values": [0.001, 0.01]  # error rates to sweep over
    }
    ```

2.  **Run the Simulation**:
    ```bash
    python main.py
    ```

3.  **Outputs**:
    Results are saved in the `sim_results/` directory, organized by geometry and configuration.
    * `counts_*.json`: Raw simulation output.
    * `*.gif`: Animated visualizations of the fluid flow.
    * `rmse_*.png`: Plots comparing noisy and ZNE-mitigated results.

## References

[1] C. A. Georgescu, M. A. Schalkers, and M. Möller, "qlbm – A quantum lattice Boltzmann software framework," *Computer Physics Communications*, vol. 315, p. 109699, 2025. [DOI: 10.1016/j.cpc.2025.109699](https://doi.org/10.1016/j.cpc.2025.109699)
