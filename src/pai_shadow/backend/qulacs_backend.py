"""Qulacs implementation of the simulation backend.

All gates are applied as explicit dense matrices (built with qiskit
conventions) so the unitaries match the qiskit backend exactly, sidestepping
qulacs' opposite rotation-sign convention. Only symmetric two-qubit rotations
(RXX/RYY/RZZ) are used, so the qulacs target-index ordering is irrelevant.

Note: qulacs and qiskit parameterise depolarizing noise differently, so noisy
results agree only approximately across backends; noiseless results match.
"""

from __future__ import annotations

from typing import List

import numpy as np
from qulacs import Observable, QuantumCircuit as QLCircuit, QuantumState
from qulacs.gate import DenseMatrix, DepolarizingNoise, TwoQubitDepolarizingNoise

from .base import Backend, NoiseSpec
from .circuit import Circuit, gate_matrix


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

    def _circuit(self, circuit: Circuit, noisy: bool = False) -> QLCircuit:
        qc = QLCircuit(circuit.num_qubits)
        ns: NoiseSpec = self.noise
        for g in circuit.gates:
            qc.add_gate(DenseMatrix(list(g.qubits), gate_matrix(g)))
            if noisy:
                if len(g.qubits) == 1 and g.name in ns.one_qubit_gates and ns.p1 > 0:
                    qc.add_gate(DepolarizingNoise(g.qubits[0], ns.p1))
                elif len(g.qubits) == 2 and g.name in ns.two_qubit_gates and ns.p2 > 0:
                    qc.add_gate(TwoQubitDepolarizingNoise(g.qubits[0], g.qubits[1], ns.p2))
        return qc

    def statevector(self, circuit: Circuit) -> np.ndarray:
        state = self._state(circuit)
        self._circuit(circuit).update_quantum_state(state)
        return state.get_vector()

    def expectation(self, circuit: Circuit, pauli: str) -> float:
        state = self._state(circuit)
        self._circuit(circuit).update_quantum_state(state)
        terms = " ".join(f"{p} {i}" for i, p in enumerate(pauli) if p != "I")
        if not terms:  # all-identity observable
            return float(np.real(np.vdot(state.get_vector(), state.get_vector())))
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
