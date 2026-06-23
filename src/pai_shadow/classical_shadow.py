"""Classical shadows with random single-qubit Pauli-basis measurements.

For single-qubit random Clifford shadows (Huang, Kueng & Preskill 2020), every
Clifford effectively measures one Pauli axis, and the single-snapshot inverse
channel reduces to a factor of 3 on the measured axis and 0 otherwise. So this
module uses the equivalent — and much cheaper — **random Pauli-basis** scheme:

  * pick an axis X/Y/Z uniformly per qubit,
  * rotate it into the computational basis with native gates (X: H, Y: S^dag H,
    Z: none) and measure,
  * the per-qubit factor is ``3 * (-1)^bit`` on the measured axis, 0 elsewhere.

The single-snapshot estimate of a Pauli string ``P = prod_q P_q`` is the product
of the per-qubit factors over the qubits where ``P_q != I`` (identity qubits give
1); averaging over snapshots is an unbiased estimate of ``<P>``. No dense gate
matrices are constructed, and many k-local observables can be estimated from the
same snapshots (what shadow spectroscopy needs).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .backend import Circuit, get_backend

_PAULI_IDX = {"X": 0, "Y": 1, "Z": 2}


class ClassicalShadow:
    """Random Pauli-basis classical shadows on a given simulation backend."""

    def __init__(self, backend=None, seed: int | None = None):
        self.backend = backend if backend is not None else get_backend("qulacs")
        self._rng = np.random.default_rng(seed)

    def _measure_basis(self, circ: Circuit, qubit: int, axis: int) -> None:
        """Append the native rotation that takes ``axis`` into the Z basis."""
        if axis == 0:        # X: H
            circ.h(qubit)
        elif axis == 1:      # Y: S^dag then H
            circ.add("SDG", [qubit])
            circ.h(qubit)
        # axis == 2 (Z): measure directly

    def snapshots(self, circuit: Circuit, n_snapshots: int) -> np.ndarray:
        """Take ``n_snapshots`` shadow snapshots of ``circuit``.

        Returns an array of shape ``(n_snapshots, num_qubits, 3)`` holding the
        per-qubit single-snapshot factors for the X, Y, Z observables (only the
        measured axis is nonzero). Feed it to :meth:`expectation` /
        :meth:`expectations`.
        """
        nq = circuit.num_qubits
        axes = self._rng.integers(0, 3, size=(n_snapshots, nq))
        factors = np.zeros((n_snapshots, nq, 3))
        for s in range(n_snapshots):
            c = Circuit(nq, list(circuit.gates), init_state=circuit.init_state)
            for q in range(nq):
                self._measure_basis(c, q, axes[s, q])
            bitstring = self.backend.sample(c, 1)[0]
            for q in range(nq):
                bit = int(bitstring[nq - 1 - q])      # qubit q is char nq-1-q
                factors[s, q, axes[s, q]] = 3.0 * (1 - 2 * bit)
        return factors

    def expectation(self, pauli: str, factors: np.ndarray) -> float:
        """Unbiased estimate of ``<pauli>`` from precomputed snapshot factors.

        ``pauli`` is a length-``num_qubits`` string over ``IXYZ`` (``pauli[i]``
        acts on qubit ``i``).
        """
        vals = np.ones(factors.shape[0])
        for q, p in enumerate(pauli):
            if p != "I":
                vals *= factors[:, q, _PAULI_IDX[p]]
        return float(vals.mean())

    def expectations(self, paulis: Sequence[str], factors: np.ndarray) -> np.ndarray:
        """Estimate many Pauli observables from the same snapshots."""
        return np.array([self.expectation(p, factors) for p in paulis])
