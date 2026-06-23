# TE-PAI Shadow Spectroscopy

Low-resource quantum **energy-gap estimation** by combining **TE-PAI**
(Time Evolution via Probabilistic Angle Interpolation) with **algorithmic shadow
spectroscopy**. TE-PAI replaces deep Trotter time-evolution circuits with shallow
randomized circuits, trading sampling overhead for circuit depth — and therefore
robustness to gate noise.

The core library (`src/pai_shadow`) is **backend-independent**: Hamiltonians and
circuits are plain data structures, and simulation runs on either **qiskit/Aer**
or **qulacs**.

## Requirements

- Python **3.10–3.12**
- [**uv**](https://docs.astral.sh/uv/) for environment and dependency management

Install uv (if needed):

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
# or with Homebrew
brew install uv
```

## Environment setup

`uv sync` creates a local virtual environment (`.venv/`), fetches a compatible
Python if necessary, and installs all dependencies pinned in `uv.lock`:

```bash
uv sync                 # runtime dependencies
uv sync --group test    # also install pytest (for the test suite)
```

Run anything inside the environment with `uv run`:

```bash
uv run python -c "import pai_shadow; print('ok')"
```

## Project structure

```
src/pai_shadow/
├── hamil.py             Pauli-sum Hamiltonians (Hamiltonian, Heisenberg_Hamil, Ising_Hamil); numpy/scipy only
├── backend/             circuit IR + simulation backends
│   ├── circuit.py         backend-independent Circuit / Gate
│   ├── base.py            Backend interface + NoiseSpec (depolarizing/bitflip/phaseflip/amp-damping)
│   ├── qiskit_backend.py
│   └── qulacs_backend.py
├── trotter.py           Hamiltonian -> first-order Trotter circuit
├── te_pai.py            TE-PAI random-circuit generator
├── classical_shadow.py  random Pauli-basis classical shadows
└── shadow_spectro.py    algorithmic shadow spectroscopy (Trotter + TE-PAI front ends)
tests/                   flat pytest suite
example/                 Jupyter notebooks (te_pai, shadow, shadow_spectroscopy)
```

## Quick start

```python
import numpy as np
from pai_shadow.hamil import Heisenberg_Hamil
from pai_shadow.backend import get_backend
from pai_shadow.trotter import trotter_circuit
from pai_shadow.te_pai import TEPAI

H = Heisenberg_Hamil(7, 1, 1, 1)          # 7-qubit Heisenberg chain
backend = get_backend("qulacs")            # or "qiskit"

# exact energy gaps (classical diagonalisation)
print(H.energy_gap()[:5])

# deterministic Trotter evolution
circ = trotter_circuit(H, t=1.0, n_steps=40)
print(backend.expectation(circ, "Z" + "I" * 6))

# TE-PAI: shallow random circuits + signed weights (unbiased estimator)
tp = TEPAI(H, delta=np.pi / 32, T=1.0, n_steps=40)
circuits, weights = tp.sample(2000)
```

Estimate an energy gap end-to-end via shadow spectroscopy:

```python
from pai_shadow.hamil import Heisenberg_Hamil
from pai_shadow.shadow_spectro import te_pai_shadow_spectroscopy, dominant_gap

H = Heisenberg_Hamil(4, 1, 1, 1)
e0, e1, g, x = H.get_ground_and_excited_state(n=1)   # gap E1 - E0
times = np.arange(50) * 0.35
freqs, spectrum = te_pai_shadow_spectroscopy(H, g + x, times, delta=np.pi / 8, M=1500)
print("recovered gap:", dominant_gap(freqs, spectrum), " exact:", e1 - e0)
```

## Running things

```bash
# tests
uv run --group test pytest tests -q

# example notebook (Trotter vs TE-PAI with error bars)
uv run jupyter lab example/shadow_spectroscopy.ipynb   # needs jupyter installed
```

## Adding dependencies

```bash
uv add <package>            # runtime dependency
uv add --group test <pkg>   # test-only dependency
```

## License

See [LICENSE](LICENSE).
