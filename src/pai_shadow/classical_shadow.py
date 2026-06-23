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

from .circuit import Circuit, Gate, native_gate

_PAULI_IDX = {"X": 0, "Y": 1, "Z": 2}


class ClassicalShadow:
    """Random Pauli-basis classical shadows (qulacs simulation)."""

    def __init__(self, seed: int | None = None):
        self._rng = np.random.default_rng(seed)

    @staticmethod
    def _rotate_to_z(state, qubit: int, axis: int) -> None:
        """Apply the native rotation taking ``axis`` into the Z basis in place."""
        if axis == 0:        # X: H
            native_gate(Gate("H", (qubit,))).update_quantum_state(state)
        elif axis == 1:      # Y: S^dag then H
            native_gate(Gate("SDG", (qubit,))).update_quantum_state(state)
            native_gate(Gate("H", (qubit,))).update_quantum_state(state)
        # axis == 2 (Z): measure directly

    def snapshots(self, circuit: Circuit, n_snapshots: int) -> np.ndarray:
        """Take ``n_snapshots`` shadow snapshots of ``circuit``.

        The (possibly deep) circuit is evolved **once**; each snapshot is then a
        cheap copy of the evolved state with random per-qubit basis rotations and
        a single measurement. Returns an array of shape
        ``(n_snapshots, num_qubits, 3)`` holding the per-qubit single-snapshot
        factors for the X, Y, Z observables (only the measured axis is nonzero).
        Feed it to :meth:`expectation` / :meth:`expectations`.
        """
        nq = circuit.num_qubits
        axes = self._rng.integers(0, 3, size=(n_snapshots, nq))
        factors = np.zeros((n_snapshots, nq, 3))
        base = circuit.evolved_state()                # apply the circuit once
        for s in range(n_snapshots):
            state = base.copy()
            row = axes[s]
            for q in range(nq):
                self._rotate_to_z(state, q, row[q])
            value = state.sampling(1)[0]
            for q in range(nq):
                bit = (value >> q) & 1                 # little-endian: qubit q
                factors[s, q, row[q]] = 3.0 * (1 - 2 * bit)
        return factors

    def snapshots_per_circuit(self, circuits: Sequence[Circuit],
                              n_shots: int = 1) -> np.ndarray:
        """``n_shots`` snapshots of each circuit; factors ``(M, n_shots, nq, 3)``.

        Used by TE-PAI shadow spectroscopy: each sampled TE-PAI circuit is
        evolved once and measured ``n_shots`` times, later combined with its
        quasiprobability weight.
        """
        nq = circuits[0].num_qubits
        out = np.empty((len(circuits), n_shots, nq, 3))
        for s, circ in enumerate(circuits):
            out[s] = self.snapshots(circ, n_shots)
        return out

    def snapshots_of_circuits(self, circuits: Sequence[Circuit]) -> np.ndarray:
        """One snapshot per circuit; returns factors ``(len(circuits), nq, 3)``."""
        return self.snapshots_per_circuit(circuits, 1)[:, 0]

    def expectation(self, pauli: str, factors: np.ndarray, weights=None) -> float:
        """Unbiased estimate of ``<pauli>`` from precomputed snapshot factors.

        ``pauli`` is a length-``num_qubits`` string over ``IXYZ`` (``pauli[i]``
        acts on qubit ``i``). Optional per-snapshot ``weights`` (e.g. TE-PAI
        quasiprobability weights) give ``mean_s( w_s * prod_q factor )``.
        """
        vals = np.ones(factors.shape[0]) if weights is None else np.array(weights, float)
        for q, p in enumerate(pauli):
            if p != "I":
                vals = vals * factors[:, q, _PAULI_IDX[p]]
        return float(vals.mean())

    def expectations(self, paulis: Sequence[str], factors: np.ndarray,
                     weights=None) -> np.ndarray:
        """Estimate many Pauli observables from the same snapshots."""
        return np.array([self.expectation(p, factors, weights) for p in paulis])
