"""TE-PAI shadow spectroscopy.

Backend-independent building blocks for estimating energy gaps via TE-PAI shadow
spectroscopy:

- :mod:`pai_shadow.hamil`    -- Pauli-sum Hamiltonians (numpy/scipy only).
- :mod:`pai_shadow.backend`  -- circuit IR + qiskit/qulacs simulation backends.
- :mod:`pai_shadow.trotter`  -- deterministic first-order Trotter circuits.
- :mod:`pai_shadow.te_pai`   -- TE-PAI random-circuit generator.

Hamiltonians are linear combinations of Pauli strings, e.g.
``[("ZZ", [0, 1], -2.0), ("X", [0], -0.1)]`` for ``H = -2 Z0 Z1 - 0.1 X0``.
"""

from .hamil import Hamiltonian, Heisenberg_Hamil, Ising_Hamil

__all__ = ["Hamiltonian", "Heisenberg_Hamil", "Ising_Hamil"]
