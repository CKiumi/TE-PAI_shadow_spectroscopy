"""TE-PAI shadow spectroscopy.

Backend-independent building blocks for estimating energy gaps via TE-PAI shadow
spectroscopy:

- :mod:`pai_shadow.hamil`    -- Pauli-sum Hamiltonians (numpy/scipy only).
- :mod:`pai_shadow.backend`  -- circuit IR + qulacs simulation backend.
- :mod:`pai_shadow.trotter`  -- deterministic first-order Trotter circuits.
- :mod:`pai_shadow.te_pai`   -- TE-PAI random-circuit generator.

Hamiltonians are linear combinations of Pauli strings, e.g.
``[("ZZ", [0, 1], -2.0), ("X", [0], -0.1)]`` for ``H = -2 Z0 Z1 - 0.1 X0``.
"""

from .hamil import Hamiltonian, Heisenberg_Hamil, Ising_Hamil
from .backend import Circuit, NoiseSpec, get_backend
from .trotter import trotter_circuit
from .te_pai import TEPAI
from .classical_shadow import ClassicalShadow
from .shadow_spectro import (
    Spectroscopy,
    dominant_gap,
    k_local_paulis,
    te_pai_shadow_spectroscopy,
    trotter_shadow_spectroscopy,
)

__all__ = [
    "Hamiltonian", "Heisenberg_Hamil", "Ising_Hamil",
    "Circuit", "NoiseSpec", "get_backend",
    "trotter_circuit", "TEPAI", "ClassicalShadow",
    "Spectroscopy", "k_local_paulis", "dominant_gap",
    "trotter_shadow_spectroscopy", "te_pai_shadow_spectroscopy",
]
