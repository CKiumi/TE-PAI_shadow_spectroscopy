"""Qiskit / Aer implementation of the simulation backend."""

from __future__ import annotations

from typing import List

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Pauli
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel, depolarizing_error, pauli_error, amplitude_damping_error,
)

from .base import Backend, NoiseSpec
from .circuit import Circuit


def _qiskit_error_1q(kind: str, p: float):
    if kind == "depolarizing":
        return depolarizing_error(p, 1)
    if kind == "bitflip":
        return pauli_error([("X", p), ("I", 1 - p)])
    if kind == "phaseflip":
        return pauli_error([("Z", p), ("I", 1 - p)])
    if kind == "amplitude_damping":
        return amplitude_damping_error(p)
    raise ValueError(kind)


def _qiskit_error_2q(kind: str, p: float):
    if kind == "depolarizing":
        return depolarizing_error(p, 2)
    e = _qiskit_error_1q(kind, p)        # apply 1q channel to each of the two qubits
    return e.tensor(e)

_APPLY = {
    "RX": lambda qc, g: qc.rx(g.param, g.qubits[0]),
    "RY": lambda qc, g: qc.ry(g.param, g.qubits[0]),
    "RZ": lambda qc, g: qc.rz(g.param, g.qubits[0]),
    "RXX": lambda qc, g: qc.rxx(g.param, g.qubits[0], g.qubits[1]),
    "RYY": lambda qc, g: qc.ryy(g.param, g.qubits[0], g.qubits[1]),
    "RZZ": lambda qc, g: qc.rzz(g.param, g.qubits[0], g.qubits[1]),
    "H": lambda qc, g: qc.h(g.qubits[0]),
    "S": lambda qc, g: qc.s(g.qubits[0]),
    "SDG": lambda qc, g: qc.sdg(g.qubits[0]),
    "X": lambda qc, g: qc.x(g.qubits[0]),
    "Y": lambda qc, g: qc.y(g.qubits[0]),
    "Z": lambda qc, g: qc.z(g.qubits[0]),
    "U": lambda qc, g: qc.unitary(g.matrix, [g.qubits[0]], label="u"),
}


class QiskitBackend(Backend):
    name = "qiskit"

    def _build(self, circuit: Circuit, measure: bool = False) -> QuantumCircuit:
        qc = QuantumCircuit(circuit.num_qubits)
        if circuit.init_state is not None:
            qc.initialize(np.asarray(circuit.init_state, dtype=complex),
                          range(circuit.num_qubits), normalize=True)
        for g in circuit.gates:
            _APPLY[g.name](qc, g)
        if measure:
            qc.measure_all()
        return qc

    def statevector(self, circuit: Circuit) -> np.ndarray:
        return Statevector(self._build(circuit)).data

    def expectation(self, circuit: Circuit, pauli: str) -> float:
        # qiskit Pauli labels are little-endian: reverse so pauli[i] -> qubit i.
        obs = Pauli(pauli[::-1])
        return float(np.real(Statevector(self._build(circuit)).expectation_value(obs)))

    def _noise_model(self) -> NoiseModel | None:
        ns: NoiseSpec = self.noise
        if ns.is_noiseless():
            return None
        nm = NoiseModel()
        names1 = [g.lower() for g in ns.one_qubit_gates]
        names2 = [g.lower() for g in ns.two_qubit_gates]
        if ns.p1 > 0 and names1:
            nm.add_all_qubit_quantum_error(_qiskit_error_1q(ns.kind, ns.p1), names1)
        if ns.p2 > 0 and names2:
            nm.add_all_qubit_quantum_error(_qiskit_error_2q(ns.kind, ns.p2), names2)
        return nm

    def sample(self, circuit: Circuit, shots: int = 1) -> List[str]:
        qc = self._build(circuit, measure=True)
        sim = AerSimulator(method="statevector", noise_model=self._noise_model())
        counts = sim.run(qc, shots=shots).result().get_counts()
        bitstrings: List[str] = []
        for bitstr, n in counts.items():
            bitstrings.extend([bitstr.replace(" ", "")] * n)
        return bitstrings
