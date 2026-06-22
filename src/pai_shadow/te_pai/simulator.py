# Written by: Chusei Kiumi, updated by Hugo PAGES
# Date: 2024-01-05

# Third-party imports
import numpy as np
from tqdm import tqdm
from qiskit import QuantumCircuit, transpile, generate_preset_pass_manager
from qiskit.circuit.library import RXXGate, RYYGate, RZZGate, RZGate, RXGate
from qiskit.quantum_info import Operator, SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import SamplerV2


def rgate(pauli, r):
    return {
        "X": RXGate(r),
        "Z": RZGate(r),
        "XX": RXXGate(r),
        "YY": RYYGate(r),
        "ZZ": RZZGate(r),
    }[pauli]


def gen_quantum_circuit(gates, nq, init_state=None, serialize: bool = False):
    circ = QuantumCircuit(nq)
    if isinstance(init_state, (np.ndarray, list)):
        circ.initialize(init_state, [i for i in range(nq)], normalize=True)

    for pauli, qubits, coef in gates:
        circ.append(rgate(pauli, coef), qubits)

    if isinstance(init_state, QuantumCircuit):
        circ = init_state.compose(circ, [i for i in range(nq)])
    if serialize:
        qasm = serialize_circuit(circ)
        del circ
        return qasm
    else:
        return circ


def sampler_measurement(circuit, shots=1):
    backend = AerSimulator(method="statevector")
    pm = generate_preset_pass_manager(backend=backend, optimization_level=2)
    isa_circuit = pm.run(circuit)
    Sampler = SamplerV2(backend)
    job = Sampler.run([isa_circuit], shots=shots)
    result = job.result()
    return result[0].data.meas.get_counts()


def get_expectation_value(circ: QuantumCircuit, obs: str) -> float:
    """get the expectation value of an observable with a given quantum circuit

    Args:
        circ (QuantumCircuit): the quantum circuit to calculate the expectation value from 
        obs (str): The observable to calculate the expectation value from 

    Returns:
        float: The expectation value 
    """
    circ_copy = circ.copy()
    circ_copy.save_expectation_value(SparsePauliOp(
        [obs]), [i for i in range(len(obs))], "0")  # type: ignore
    sim = AerSimulator(method="statevector")
    try:
        data = sim.run(circ_copy).result().data()
    except Exception as e:
        qct = transpile(circ_copy, sim)
        data = sim.run(qct).result().data()
    return data["0"]


def Circuit_to_matrix(circ):
    # backend = AerSimulator(method='unitary')
    unitary = Operator(circ).data
    return unitary


def get_bit_string(circ, shots=1):
    sim = AerSimulator(method="statevector")
    counts = sim.run(circ, shots=1).result().get_counts()
    return list(counts.keys())[0]


def C_measurements(C, verbose=True, shots=1):
    bit_string = []
    for circuit in tqdm(C, ascii=True, desc="sampling", disable=not verbose):
        sim = AerSimulator(method="statevector")
        counts = sim.run(circuit, shots=1).result().get_counts()
        bit_string.append(list(counts.keys())[0])  # type: ignore
    return bit_string


def serialize_circuit(circuit: QuantumCircuit) -> str:
    """Serialize a QuantumCircuit into JSON."""
    from qiskit.qasm2 import dumps
    qasm_string = dumps(circuit)  # Convert circuit to OpenQASM 2.0
    gate_definitions = """
    gate rxx(theta) a, b { cx a, b; rz(theta) b; cx a, b; }
    gate ryy(theta) a, b { ry(-pi/2) a; ry(-pi/2) b; cx a, b; rz(theta) b; cx a, b; ry(pi/2) a; ry(pi/2) b; }
    gate rzz(theta) a, b { cx a, b; rz(theta) b; cx a, b; }
    """
    qasm_lines = qasm_string.split("\n")
    qasm_lines.insert(2, gate_definitions.strip())
    qasm = "".join(qasm_lines)
    return qasm
