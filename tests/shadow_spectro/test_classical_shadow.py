from pai_shadow.tools_box.quantum_tools import get_expectation_value, get_q_local_Pauli, serialize_circuit, qiskit_get_last_single_qubit_gates, qiskit_is_transpiled_for_backend
from tqdm import tqdm
from qiskit import QuantumCircuit
from qiskit.circuit.random import random_circuit, random_clifford_circuit
import numpy as np
from pai_shadow.shadow_spectro.classical_shadow import ClassicalShadow
import sys
import os
folder = os.path.dirname(os.path.dirname(
    os.path.abspath(os.path.dirname(sys.argv[0]))))
sys.path.append(folder)


def test_shadow_expectation_value():
    shadow = ClassicalShadow()
    obs = get_q_local_Pauli(4, 3)
    circ = random_clifford_circuit(num_qubits=4, num_gates=6)
    index = np.random.randint(0, len(obs))
    shadow_res = 0
    for _ in tqdm(range(2000)):
        cliff, bit_string = shadow.classical_shadow(circ)
        shadow_res += shadow.get_expectation_value(
            obs[index], cliff, bit_string)
    ideal = get_expectation_value(circ, "".join(obs[index]))
    shadow_res /= 2000
    error = np.abs(shadow_res-ideal)**2
    print(error)
    assert (error < 0.1), "Error Distance between the ideal value and the expectation value from classical shadow is >0.01"


def test_shadow_density_matrix_bell_state():
    # bell state preparation
    circuit = QuantumCircuit(2)
    circuit.h(0)
    # Controlled-NOT. The controlled and target bit are 0th and 1st one, respectively.
    circuit.cx(0, 1)
    shadow = ClassicalShadow()
    ideal_res = np.array([[0.5, 0, 0, 0.5], [0, 0, 0, 0],
                         [0, 0, 0, 0], [0.5, 0, 0, 0.5]])
    rho = np.zeros_like(ideal_res, dtype="complex128")
    rho2 = np.zeros_like(ideal_res, dtype="complex128")
    for i in tqdm(range(5000)):
        cliff, bit_string = shadow.classical_shadow(circuit)
        rho += shadow.snapshot_density_matrix(cliff, bit_string)
        rho2 += shadow.classical_shadow(circuit, density_matrix=True)
    rho = rho/5000
    rho2 = rho2/5000
    diff = ideal_res-rho
    distance = np.sqrt(np.trace(diff.conj().transpose()@diff))
    diff = ideal_res-rho2
    distance2 = np.sqrt(np.trace(diff.conj().transpose()@diff))
    assert np.abs(distance) < 0.15 and np.abs(
        distance2) < 0.15, "Error Distance between the ideal value and the density matrix from classical shadow is >0.15"


def test_arg_type():
    shadow = ClassicalShadow()
    qc = random_circuit(4, 4)
    import time
    try:
        assert isinstance(shadow.random_clifford_gate(),
                          str), "random_clifford_gate should return a string"
        assert len(shadow.random_clifford_gate()
                   ), "random_clifford_gate should return a string of length 3"
    except Exception as e:
        assert False, e
    try:
        qc = random_circuit(4, 4)
        qc_test = qc.copy()
        cliff = shadow.add_random_clifford(qc)
        assert isinstance(cliff, list), "cliff should be a list"
        assert isinstance(cliff[0], str),   "cliff should be a list of string"
        assert len(cliff) == 4,   "len(cliff) should be equal to number of qubits"
        assert len(cliff[0]) == 3,   "len(cliff[0]) should be equal to 3"
        assert (qc != qc_test), "The input circuit should be modified"
        qc.remove_final_measurements()
        data = qiskit_get_last_single_qubit_gates(qc)
        # for n,instruction in enumerate(data):
        #     assert (np.allclose(instruction.params[0],np.linalg.multi_dot([shadow.gate_set[cliff[n][i]]for i in range(len(cliff[n]))]), atol=0.01)), "The last gate of each qubit in the circuit should be equal to the return clifford gate list"
        qc = random_circuit(4, 4)
        qc_test = qc.copy()
        cliff, circuit = shadow.add_random_clifford(qc, copy=True)
        assert isinstance(cliff, list), "cliff should be a list"
        assert isinstance(cliff[0], str),   "cliff should be a list of string"
        assert len(cliff) == 4,   "len(cliff) should be equal to number of qubits"
        assert len(cliff[0]) == 3,   "len(cliff[0]) should be equal to 3"
        assert (qc == qc_test), "The input circuit should not be modified"
        assert (
            circuit != qc), "The output circuit should be different from the input circuit"
        circuit.remove_final_measurements()
        data = qiskit_get_last_single_qubit_gates(circuit)
        # for n,instruction in enumerate(data):
        #     assert (np.allclose(np.round(instruction.params[0],2),np.round(np.linalg.multi_dot([shadow.gate_set[cliff[n][i]]for i in range(len(cliff[n]))]),2), atol=0.01)), "The last gate in circuit should be equal to the return clifford gate list"
    except Exception as e:
        assert False, e
    try:
        qc = random_clifford_circuit(4, 4, gates=[
            "i", "x", "y", "z", "h"])
        qasm = serialize_circuit(qc)
        cliff, bit_string = shadow.classical_shadow(qasm)
        assert isinstance(cliff, list), "cliff shoulf be a list"
        assert isinstance(cliff[0], str), "cliff shoulf be a list of tring"
        assert len(cliff) == 4, "len(cliff) should be equal to number of qubits"
        assert len(cliff[0]) == 3, "len(cliff[0]) should be equal to 3"
        assert isinstance(bit_string, str), "bit_string should be a string"
        assert len(
            bit_string) == 4, "len(bit_string) should be equal to number of qubits"
    except Exception as e:
        assert False, e
    try:
        from qiskit_ibm_runtime.fake_provider import FakeManilaV2
        backend = FakeManilaV2()
        qc = random_circuit(4, 6)
        qc_test = qc.copy()
        cliff, circuit = shadow.add_random_clifford(
            qc, copy=True, backend=backend)
        assert isinstance(cliff, list), "cliff should be a list"
        assert isinstance(cliff[0], str),   "cliff should be a list of string"
        assert len(cliff) == 4,   "len(cliff) should be equal to number of qubits"
        assert len(cliff[0]) == 3,   "len(cliff[0]) should be equal to 3"
        assert (qc == qc_test), "The input circuit should not be modified"
        assert (
            circuit != qc), "The output circuit should be different from the input circuit"
        assert qiskit_is_transpiled_for_backend(
            circuit, backend), "The output circuit should be transpiled "
    except Exception as e:
        assert False, e
