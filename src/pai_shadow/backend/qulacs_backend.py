"""Qulacs implementation of the simulation backend.

Gates use qulacs' native operations. qulacs' rotation gates use the opposite
sign convention to qiskit (qulacs ``RX(i, t) = exp(+i t/2 X)``), so we negate
the angle to match the qiskit convention ``RX(t) = exp(-i t/2 X)``. Two-qubit
Pauli rotations (RXX/RYY/RZZ) map onto ``PauliRotation`` with the same negation.
Custom single-qubit unitaries (``U``, e.g. classical-shadow Cliffords) are the
only case that needs an explicit ``DenseMatrix``.

Note: qulacs and qiskit parameterise depolarizing noise differently, so noisy
results agree only approximately across backends; noiseless results match.
"""

from __future__ import annotations

from typing import List

import numpy as np
from qulacs import DensityMatrix, Observable, QuantumCircuit as QLCircuit, QuantumState
from qulacs.gate import (
    RX, RY, RZ, H, S, Sdag, X, Y, Z,
    DenseMatrix, PauliRotation, DepolarizingNoise, TwoQubitDepolarizingNoise,
    BitFlipNoise, DephasingNoise, AmplitudeDampingNoise,
)

from .base import Backend, NoiseSpec
from .circuit import Circuit, ONE_QUBIT_ROTATIONS, TWO_QUBIT_ROTATIONS

_ROT_1Q = {"RX": RX, "RY": RY, "RZ": RZ}
_FIXED_1Q = {"H": H, "S": S, "SDG": Sdag, "X": X, "Y": Y, "Z": Z}
_PAULI_ID = {"X": 1, "Y": 2, "Z": 3}
_NOISE_1Q = {
    "depolarizing": DepolarizingNoise,
    "bitflip": BitFlipNoise,
    "phaseflip": DephasingNoise,
    "amplitude_damping": AmplitudeDampingNoise,
}


def _noise_gates(kind, qubits, p):
    """qulacs noise gate(s) to apply after a gate on ``qubits`` with rate ``p``."""
    if len(qubits) == 2:
        if kind == "depolarizing":
            return [TwoQubitDepolarizingNoise(qubits[0], qubits[1], p)]
        return [_NOISE_1Q[kind](qubits[0], p), _NOISE_1Q[kind](qubits[1], p)]
    return [_NOISE_1Q[kind](qubits[0], p)]


def _native_gate(g):
    """Build the qulacs gate for an IR Gate (qiskit sign conventions)."""
    name = g.name
    if name in ONE_QUBIT_ROTATIONS:
        return _ROT_1Q[name](g.qubits[0], -g.param)          # negate to match qiskit
    if name in TWO_QUBIT_ROTATIONS:
        pid = _PAULI_ID[name[1]]
        return PauliRotation(list(g.qubits), [pid, pid], -g.param)
    if name == "U":
        return DenseMatrix(g.qubits[0], np.asarray(g.matrix, dtype=complex))
    return _FIXED_1Q[name](g.qubits[0])


class QulacsBackend(Backend):
    name = "qulacs"

    def _state(self, circuit: Circuit) -> QuantumState:
        state = QuantumState(circuit.num_qubits)
        if circuit.init_state is not None:
            vec = np.asarray(circuit.init_state, dtype=complex)
            vec = vec / np.linalg.norm(vec)
            state.load(vec)
        else:
            state.set_zero_state()
        return state

    def _density(self, circuit: Circuit) -> DensityMatrix:
        dm = DensityMatrix(circuit.num_qubits)
        if circuit.init_state is not None:
            vec = np.asarray(circuit.init_state, dtype=complex)
            dm.load(vec / np.linalg.norm(vec))
        else:
            dm.set_zero_state()
        return dm

    def _circuit(self, circuit: Circuit, noisy: bool = False) -> QLCircuit:
        qc = QLCircuit(circuit.num_qubits)
        ns: NoiseSpec = self.noise
        for g in circuit.gates:
            qc.add_gate(_native_gate(g))
            if noisy:
                if len(g.qubits) == 1 and g.name in ns.one_qubit_gates and ns.p1 > 0:
                    for ng in _noise_gates(ns.kind, g.qubits, ns.p1):
                        qc.add_gate(ng)
                elif len(g.qubits) == 2 and g.name in ns.two_qubit_gates and ns.p2 > 0:
                    for ng in _noise_gates(ns.kind, g.qubits, ns.p2):
                        qc.add_gate(ng)
        return qc

    def statevector(self, circuit: Circuit) -> np.ndarray:
        state = self._state(circuit)
        self._circuit(circuit).update_quantum_state(state)
        return state.get_vector()

    def expectation(self, circuit: Circuit, pauli: str) -> float:
        """Expectation of ``pauli``. With noise, the exact noisy value is
        computed via density-matrix evolution (no sampling)."""
        terms = " ".join(f"{p} {i}" for i, p in enumerate(pauli) if p != "I")
        if not terms:  # all-identity observable
            return 1.0
        if self.noise.is_noiseless():
            state = self._state(circuit)
            self._circuit(circuit).update_quantum_state(state)
        else:
            state = self._density(circuit)
            self._circuit(circuit, noisy=True).update_quantum_state(state)
        obs = Observable(circuit.num_qubits)
        obs.add_operator(1.0, terms)
        return float(np.real(obs.get_expectation_value(state)))

    def sample(self, circuit: Circuit, shots: int = 1) -> List[str]:
        n = circuit.num_qubits
        if self.noise.is_noiseless():
            state = self._state(circuit)
            self._circuit(circuit).update_quantum_state(state)
            ints = state.sampling(shots)
            return [self._int_to_bitstring(s, n) for s in ints]
        # Noisy: each shot is an independent stochastic realisation.
        qc = self._circuit(circuit, noisy=True)
        out: List[str] = []
        for _ in range(shots):
            state = self._state(circuit)
            qc.update_quantum_state(state)
            out.append(self._int_to_bitstring(state.sampling(1)[0], n))
        return out

    @staticmethod
    def _int_to_bitstring(value: int, n: int) -> str:
        # qubit n-1 left-most, qubit 0 right-most (matches the qiskit backend).
        return "".join(str((value >> i) & 1) for i in reversed(range(n)))
