"""Classical shadows with random single-qubit Clifford measurements.

Following Huang, Kueng & Preskill (2020): each snapshot applies an independent
random single-qubit Clifford to every qubit and measures in the computational
basis. The single-qubit inverse channel reconstructs

    rho_hat_q = 3 U_q^dagger |b_q><b_q| U_q - I ,

and the single-snapshot estimate of a Pauli string ``P = prod_q P_q`` is the
product of ``tr(rho_hat_q P_q)`` over the qubits where ``P_q != I`` (identity
qubits contribute 1). Averaging over snapshots gives an unbiased estimate of
``<P>``.

Because the per-qubit factor depends only on the sampled Clifford index and the
measured bit, all factors are precomputed into a small table, so estimating many
k-local Pauli observables from the same snapshots is cheap (this is what shadow
spectroscopy needs).
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from .backend import Circuit, get_backend

# Single-qubit operators.
_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_S = np.array([[1, 0], [0, 1j]], dtype=complex)
_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
_V = _H @ _S @ _H @ _S
_W = _V @ _V
_KET = (np.array([[1, 0], [0, 0]], dtype=complex),   # |0><0|
        np.array([[0, 0], [0, 1]], dtype=complex))   # |1><1|

_PAULI_IDX = {"X": 0, "Y": 1, "Z": 2}
_PAULIS = (_X, _Y, _Z)


def _clifford_set() -> np.ndarray:
    """The 24 single-qubit Clifford matrices (a 3-design)."""
    g = {"X": _X, "Y": _Y, "Z": _Z, "I": _I, "S": _S, "H": _H, "V": _V, "W": _W}
    labels = ["III", "XII", "YII", "ZII", "VII", "VXI", "VYI", "VZI",
              "WXI", "WYI", "WZI", "HXI", "HYI", "HZI", "HII", "HVI",
              "HVX", "HVY", "HVZ", "HWI", "HWX", "HWY", "HWZ", "WII"]
    return np.array([g[a] @ g[b] @ g[c] for a, b, c in labels])


class ClassicalShadow:
    """Random single-qubit Clifford classical shadows on a given backend."""

    def __init__(self, backend=None, seed: int | None = None):
        self.backend = backend if backend is not None else get_backend("qulacs")
        self._cliffords = _clifford_set()                # (24, 2, 2)
        self._rng = np.random.default_rng(seed)
        # factor[c, b, p] = tr((3 U_c^dag |b><b| U_c - I) sigma_p)
        self._factor = np.empty((24, 2, 3))
        for c in range(24):
            U = self._cliffords[c]
            for b in range(2):
                rho = 3 * U.conj().T @ _KET[b] @ U - _I
                for p, sig in enumerate(_PAULIS):
                    self._factor[c, b, p] = np.trace(rho @ sig).real

    def snapshots(self, circuit: Circuit, n_snapshots: int) -> np.ndarray:
        """Take ``n_snapshots`` shadow snapshots of ``circuit``.

        Returns an array of shape ``(n_snapshots, num_qubits, 3)`` holding the
        per-qubit single-snapshot factors for the X, Y, Z observables. Feed it to
        :meth:`expectation` / :meth:`expectations`.
        """
        nq = circuit.num_qubits
        idx = self._rng.integers(0, 24, size=(n_snapshots, nq))
        bits = np.empty((n_snapshots, nq), dtype=int)
        for s in range(n_snapshots):
            c = Circuit(nq, list(circuit.gates), init_state=circuit.init_state)
            for q in range(nq):
                c.unitary(q, self._cliffords[idx[s, q]])
            bitstring = self.backend.sample(c, 1)[0]
            for q in range(nq):
                bits[s, q] = int(bitstring[nq - 1 - q])  # qubit q is char nq-1-q
        return self._factor[idx, bits]                    # (n, nq, 3)

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
