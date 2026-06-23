"""Backend-independent circuit intermediate representation (IR).

A :class:`Circuit` is a flat list of :class:`Gate` operations on a fixed number
of qubits, with an optional initial state vector. It carries no dependency on
any simulator; the qulacs backend translates it to its native representation.

Gate conventions follow the standard sign convention:
    RX(t) = exp(-i t/2 X),  RZ(t) = exp(-i t/2 Z),
    RXX(t) = exp(-i t/2 X⊗X),  etc.
The two-qubit rotation gates (RXX/RYY/RZZ) are symmetric, so the order of their
two qubit arguments is irrelevant. Qubit ``i`` of the circuit maps to native
qubit ``i``; amplitudes use little-endian indexing (qubit 0 is the
least-significant bit).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

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
