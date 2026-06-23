"""Backend-independent circuit intermediate representation (IR).

A :class:`Circuit` is a flat list of :class:`Gate` operations on a fixed number
of qubits, with an optional initial state vector. It carries no dependency on
any simulator; concrete backends (qiskit, qulacs) translate it to their native
representation.

Gate conventions follow qiskit:
    RX(t) = exp(-i t/2 X),  RZ(t) = exp(-i t/2 Z),
    RXX(t) = exp(-i t/2 X⊗X),  etc.
The two-qubit rotation gates (RXX/RYY/RZZ) are symmetric, so the order of their
two qubit arguments is irrelevant. Qubit ``i`` of the circuit maps to native
qubit ``i``; both supported backends use little-endian amplitude indexing
(qubit 0 is the least-significant bit).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import List, Optional, Sequence, Tuple

import numpy as np

# Fixed single-qubit matrices.
_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
_S = np.array([[1, 0], [0, 1j]], dtype=complex)
_SDG = np.array([[1, 0], [0, -1j]], dtype=complex)

_FIXED = {"I": _I, "X": _X, "Y": _Y, "Z": _Z, "H": _H, "S": _S, "SDG": _SDG}
_PAULI = {"X": _X, "Y": _Y, "Z": _Z}
ONE_QUBIT_ROTATIONS = {"RX", "RY", "RZ"}
TWO_QUBIT_ROTATIONS = {"RXX", "RYY", "RZZ"}


@dataclass
class Gate:
    """A single operation: ``name`` acting on ``qubits`` with optional ``param``.

    For ``name == "U"`` an explicit 2x2 ``matrix`` (custom single-qubit unitary,
    e.g. a classical-shadow Clifford) must be supplied.
    """

    name: str
    qubits: Tuple[int, ...]
    param: Optional[float] = None
    matrix: Optional[np.ndarray] = None


@lru_cache(maxsize=None)
def _matrix_cached(name: str, param: Optional[float]) -> np.ndarray:
    """Cached matrix for parameter-only gates (keyed by name + angle).

    TE-PAI uses a tiny set of angles ({+/-delta, pi}) repeated across millions of
    gates, so caching here avoids recomputing the same cos/sin matrices.
    """
    if name in _FIXED:
        return _FIXED[name]
    if name in ONE_QUBIT_ROTATIONS:
        P = {"RX": _X, "RY": _Y, "RZ": _Z}[name]
        return np.cos(param / 2) * _I - 1j * np.sin(param / 2) * P
    if name in TWO_QUBIT_ROTATIONS:
        PP = np.kron(_PAULI[name[1]], _PAULI[name[1]])
        return np.cos(param / 2) * np.eye(4, dtype=complex) - 1j * np.sin(param / 2) * PP
    raise ValueError(f"Unknown gate name {name!r}.")


def gate_matrix(gate: Gate) -> np.ndarray:
    """Return the dense unitary matrix of ``gate`` (qiskit conventions)."""
    if gate.name == "U":
        if gate.matrix is None:
            raise ValueError("Gate 'U' requires an explicit matrix.")
        return np.asarray(gate.matrix, dtype=complex)
    return _matrix_cached(gate.name, gate.param)


@dataclass
class Circuit:
    """A backend-independent quantum circuit."""

    num_qubits: int
    gates: List[Gate] = field(default_factory=list)
    init_state: Optional[np.ndarray] = None  # length-2**n statevector, optional

    # -- generic builder ------------------------------------------------- #
    def add(self, name: str, qubits: Sequence[int], param: float | None = None,
            matrix: np.ndarray | None = None) -> "Circuit":
        self.gates.append(Gate(name, tuple(qubits), param, matrix))
        return self

    # -- convenience builders ------------------------------------------- #
    def rx(self, q: int, theta: float) -> "Circuit": return self.add("RX", [q], theta)
    def ry(self, q: int, theta: float) -> "Circuit": return self.add("RY", [q], theta)
    def rz(self, q: int, theta: float) -> "Circuit": return self.add("RZ", [q], theta)
    def rxx(self, a: int, b: int, theta: float) -> "Circuit": return self.add("RXX", [a, b], theta)
    def ryy(self, a: int, b: int, theta: float) -> "Circuit": return self.add("RYY", [a, b], theta)
    def rzz(self, a: int, b: int, theta: float) -> "Circuit": return self.add("RZZ", [a, b], theta)
    def h(self, q: int) -> "Circuit": return self.add("H", [q])
    def s(self, q: int) -> "Circuit": return self.add("S", [q])
    def x(self, q: int) -> "Circuit": return self.add("X", [q])
    def y(self, q: int) -> "Circuit": return self.add("Y", [q])
    def z(self, q: int) -> "Circuit": return self.add("Z", [q])
    def unitary(self, q: int, matrix: np.ndarray) -> "Circuit": return self.add("U", [q], matrix=matrix)

    def rotation(self, pauli: str, qubits: Sequence[int], theta: float) -> "Circuit":
        """Append a Pauli rotation from a Pauli label: 'X'->RX, 'ZZ'->RZZ, ..."""
        return self.add("R" + pauli.upper(), qubits, theta)

    def __len__(self) -> int:
        return len(self.gates)

    def depth(self) -> int:
        """Circuit depth (longest path of gates sharing qubits)."""
        layer = [0] * self.num_qubits
        for g in self.gates:
            d = max(layer[q] for q in g.qubits) + 1
            for q in g.qubits:
                layer[q] = d
        return max(layer) if self.gates else 0
