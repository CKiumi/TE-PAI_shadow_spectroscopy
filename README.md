# TE_PAI Shadow Spectroscopy

This repository provides a framework for quantum simulation and shadow spectroscopy, centered around the `Te_pai_shadow` class. The project is designed for extensibility and reproducibility, leveraging modern Python packaging and dependency management with [Poetry](https://python-poetry.org/).

## Project Structure

- **main.py**: Entry point to run simulations using the `Te_pai_shadow` class.
- **post_processing.py**: Script for post-processing and analyzing simulation data.
- **te_pai_shadow/**: Contains the main `Te_pai_shadow` implementation and related utilities.
- **te_pai/**: Includes the Trotterization logic and related simulation tools.
- **Shadow_Spectro/**: Implements shadow spectroscopy methods.
- **Hamiltonian/**, **Hardware_simulation/**, **tools_box/**: Supporting modules for Hamiltonian definitions, hardware simulation, and utility functions.

## Main Components

### Te_pai_shadow

The core class of this workspace, located in `te_pai_shadow/te_pai_shadow_spectro.py`, orchestrates quantum simulations and shadow spectroscopy. It requires:

- **TE-PAI**: For Time evolution using Probabilistic angle interpolation of spin chain Hamiltonian, implemented in `TE_PAI/TE_PAI.py`.
- **ShadowSpectro**: For shadow spectroscopy, implemented in `Shadow_Spectro/ShadowSpectro.py`.


## Installation

This project uses [Poetry](https://python-poetry.org/) for dependency management and packaging. To install all requirements:

```bash
# Install Poetry if you haven't already
pip install poetry

# Install dependencies
poetry install
```

## Usage

To run a simulation using the main class:

```bash
poetry run python main.py
```


This will execute the main workflow using `Te_pai_shadow`, which internally utilizes Trotterization and shadow spectroscopy.

## Post-processing

After running simulations, you can analyze and visualize results using:

```bash
poetry run python post_processing.py
```

## Requirements

All dependencies are specified in `pyproject.toml` and managed via Poetry. Typical requirements include:

    python = "^3.10"
    matplotlib = "3.9.2"
    multiprocess = "0.70.17"
    numba = "0.60.0"
    qiskit = "^1.2.4"
    qiskit-aer = "0.15.1"
    qiskit-ibm-runtime = "0.31.0"
    scipy = "1.14.1"
    tqdm = "4.66.5"
    statsmodels = "^0.14.4"
To add new dependencies, use:

```bash
poetry add <package-name>
```

## Testing

Unit tests are provided in the `TEST/` directory. To run tests:

```bash
poetry run pytest
```

## License

See [LICENSE](LICENSE) for details.

---
For further details, refer to the source code and docstrings within each module.
