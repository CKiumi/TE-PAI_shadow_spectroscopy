"""Deterministic first-order Trotter time-evolution circuits.

Builds a backend-independent :class:`~pai_shadow.backend.circuit.Circuit` that
approximates ``exp(-i H t)`` for a :class:`~pai_shadow.hamil.Hamiltonian` using
a first-order product formula.

For a term ``coef * P`` of the Hamiltonian, one Trotter step of duration ``dt``
contributes the rotation ``exp(-i coef dt P)``. With the qiskit rotation
convention ``R_P(theta) = exp(-i theta/2 P)`` this is angle ``theta = 2 coef dt``.
"""

from __future__ import annotations

import numpy as np

from .backend.circuit import Circuit
from .hamil import Hamiltonian


def trotter_circuit(
    hamil: Hamiltonian,
    t: float,
    n_steps: int,
    init_state: np.ndarray | None = None,
) -> Circuit:
    """Return a first-order Trotterized circuit approximating ``exp(-i H t)``.

    Parameters
    ----------
    hamil:
        The Hamiltonian to evolve under.
    t:
        Total evolution time.
    n_steps:
        Number of Trotter steps (step size ``dt = t / n_steps``).
    init_state:
        Optional initial state vector (little-endian) prepended to the circuit.
    """
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1.")

    circ = Circuit(hamil.nqubits, init_state=init_state)
    if t == 0:
        return circ

    dt = t / n_steps
    for step in range(n_steps):
        # Midpoint time of the step (matters only for time-dependent coefficients).
        for pauli, qubits, coef in hamil.get_term((step + 0.5) * dt):
            circ.rotation(pauli, qubits, 2.0 * float(np.real(coef)) * dt)
    return circ
