# Written by: Hugo PAGES 
# Date: 2024-01-05

# Standard library imports
from typing import Union

# Third-party imports
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UnitaryGate, RXXGate, RYYGate, RZZGate
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit.qasm2 import loads, CustomInstruction
import secrets
class GateConstructor:
    def __init__(self, matrix):
        self.matrix = matrix

    def __call__(self):
        return UnitaryGate(self.matrix)

# Standard library imports
import re
import time
import multiprocessing as mp
from dataclasses import dataclass

# Third-party imports
import numpy as np
from tqdm import tqdm
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import RXXGate, RYYGate, RZZGate, RZGate, RXGate

# Local application imports
from pai_shadow.te_pai import pai, sampling, simulator


@dataclass
class fast_TE_PAI_shadow:
    """
    TE-PAI Circuit Generator for Time-Evolution Probabilistic Angle Interpolation (TE-PAI)
    of quantum systems under Hamiltonian dynamics.

    This class constructs randomized circuits approximating Hamiltonian evolution using
    a probabilistic interpolation method that achieves angle precision within a defined
    threshold. It allows efficient simulation via circuit generation, serialization, and
    estimation of observable expectation values.
    """
    def create_gate_function(self, matrix):
        return GateConstructor(matrix)
    
    def __init__(self, hamil, numQs, delta, T,
                 trotter_steps=1.e-4, PAI_error=0.05, N_trotter_max=8000, M_sample_max=500,
                 init_state=None, backend=None, shadow_size=1):
        """
        Initialize the TE_PAI class.

        Args:
            hamil (Hamiltonian): Hamiltonian to be evolved under.
            numQs (int): Number of qubits.
            delta (float): Angular precision used in PAI.
            T (float): Total time evolution.
            trotter_steps (float): Time step.
            PAI_error (float): Target precision error for probabilistic angle interpolation.
            N_trotter_max (int): Maximum number of Trotter steps.
            M_sample_max (int): Maximum number of sampled circuits.
            init_state (Union[np.ndarray, QuantumCircuit], optional): Initial quantum state.
            serialize (bool, optional): Whether to serialize circuits to QASM. Defaults to False.
        """
        self.nq, self.delta, self.T = numQs, delta, T
        self.N=N_trotter_max
        
        self.N = int(T / trotter_steps)
        if self.N == 0:
            self.N = 1
        if self.N > N_trotter_max:
            self.N = N_trotter_max

        steps = np.linspace(0, T, self.N, endpoint=True)
        self.L = len(hamil)
        self.angles = np.array(
            [2 * np.abs(hamil.coefs(t)) * T / self.N for t in steps])
        self.gamma = np.prod([pai.gamma(angle, delta)
                             for angle in self.angles])

        self.probs = np.array([pai.prob_list(angle, delta)
                              for angle in self.angles])
        self.terms = np.array([hamil.get_term(t) for t in steps], dtype=object)

        self.overhead = self.gamma
        
        self.M_sample = int((self.overhead**2/(PAI_error**2)))

        
        if self.M_sample > M_sample_max:
            self.M_sample = M_sample_max
        if self.M_sample < 60:
            self.M_sample = 60
            
        self.num_processes = min(30, int(mp.cpu_count() * 0.30))
        self.backend=backend
        print("overhead:", self.overhead)
        print("M_sample:", self.M_sample)
        print("Trot :", self.N)

        self.init_state = init_state
        self.bitstring_matrix0 = np.array([[1, 0], [0, 0]])
        self.bitstring_matrix1 = np.array([[0, 0], [0, 1]])
        self.X = np.array([[0, 1],  [1, 0]])
        self.Y = np.array([[0, -1j], [1j, 0]])
        self.Z = np.array([[1, 0],  [0, -1]])
        self.I = np.array([[1, 0],  [0, 1]])
        self.S = np.array([[1, 0],  [0, 1j]])
        self.H = np.array([[1/np.sqrt(2), 1/np.sqrt(2)],
                          [1/np.sqrt(2), -1/np.sqrt(2)]])
        self.V = self.H@self.S@self.H@self.S
        self.W = self.V@self.V
        self.gate_set = {"X": self.X, "Y": self.Y, "Z": self.Z, "I": self.I, "S": self.S,
                         "H": self.H, "V": self.V, "W": self.W}
        self.Clifford_Gate_set = [
            "III", "XII", "YII", "ZII", "VII", "VXI", "VYI", "VZI",
            "WXI", "WYI", "WZI", "HXI", "HYI", "HZI", "HII",
            "HVI", "HVX", "HVY", "HVZ", "HWI", "HWX",
            "HWY", "HWZ", "WII"]
        self.precompiled_cliffords = {}

        for label in self.Clifford_Gate_set:
            matrix = np.linalg.multi_dot([self.gate_set[gate] for gate in label])
            unitary = UnitaryGate(matrix, label=label)
            qc = QuantumCircuit(1)
            qc.append(unitary, [0])
            transpiled =  transpile(qc, basis_gates=self.backend.configuration().basis_gates, optimization_level=3)
            self.precompiled_cliffords[label] = transpiled
        
        
        
        
         # Extracting the gate to apply
        rxx_custom = CustomInstruction(
        name="rxx", num_params=1, num_qubits=2, builtin=False, constructor=RXXGate)
        ryy_custom = CustomInstruction(
            name="ryy", num_params=1, num_qubits=2, builtin=False,  constructor=RYYGate)
        rzz_custom = CustomInstruction(
            name="rzz", num_params=1, num_qubits=2, builtin=False, constructor=RZZGate)
        self.custom_instruction_list=[rxx_custom,ryy_custom,rzz_custom]
        for gate in self.Clifford_Gate_set:
            gate_matrix=self.gate_set[gate[0]]@self.gate_set[gate[1]]@self.gate_set[gate[2]]
            gate_constructor = self.create_gate_function(gate_matrix)
            gate_instruction=CustomInstruction(
                name=str.lower(gate), num_params=0, num_qubits=1, builtin=False, constructor=gate_constructor)
            self.custom_instruction_list.append(gate_instruction)
        
        
        self.serialize=False
        self.chunksize = max(1, self.M_sample // (self.num_processes * 6))
        self.shadow_size=shadow_size
        # self.layout_15=[15,19,35,34,33,39,53,54,55,59,75,74,73,75,71]
        if isinstance(self.init_state, (np.ndarray, list)):
            init_circ = QuantumCircuit(self.nq)
            init_circ.initialize(self.init_state, list(range(self.nq)), normalize=True)
            self._cached_init_circ = transpile(init_circ, optimization_level=3, basis_gates=self.backend.configuration().basis_gates)

        elif isinstance(self.init_state, QuantumCircuit):
            self._cached_init_circ = transpile( self.init_state.copy(),optimization_level=3, basis_gates=self.backend.configuration().basis_gates)
            
    def __del__(self):
        """
        Explicitly clears class attributes to help with garbage collection.
        """
        self.angles = None
        self.probs = None
        self.terms = None
        self.gamma = None
        self.GAMMA = None
        self.M_sample = 0
        self.init_state = None
        self.serialize = None

    def gen_te_pai_circuits(self):
        """
        Generate TE-PAI quantum circuits via randomized sampling of angle terms.
        This method utilizes multiprocessing to efficiently generate circuits
        """
        from multiprocessing.pool import ThreadPool


        t = time.time()
        index = sampling.batch_sampling(self.probs, self.M_sample)
        res = []
        
        with mp.Pool(self.num_processes) as pool:
            res += pool.map(self.gen_cir_from_index, index, chunksize=self.chunksize)
        print("time to generate from index :", time.time() - t)
        gates_array, self.GAMMA = zip(*res)
        self.GAMMA = np.array(self.GAMMA)

        results = []
        t2 = time.time()
        with mp.Pool(self.num_processes) as pool:
            results = pool.map(self.gen_quantum_circuit, gates_array, chunksize=self.chunksize)
        snapshot_Clifford_array, Circuits , depth = zip(*results)
        self.TE_PAI_Circuits=[]
        for shadow_circ in Circuits :
            for circ in shadow_circ:
                self.TE_PAI_Circuits.append(circ)
                

        
        self.snapshot_Clifford_array = list(snapshot_Clifford_array)
        print("Average depth circuits :", np.mean(depth) )
        print("time to generate from gates array:", time.time() - t2)
        print("Total time:", time.time() - t)

    def get_expectation_value(self, observable, multiprocessing=True):
        """
        Compute the expectation value of an observable from the TE-PAI circuit ensemble.

        Args:
            observable (PauliSumOp or compatible): Observable for which to compute expectation.
            multiprocessing (bool): Whether to parallelize the evaluation.

        Returns:
            tuple: Mean and standard deviation of measured observable.
        """
        self.observable = observable
        if multiprocessing:
            with mp.Pool(processes=self.num_processes) as pool:
                results = pool.map(self.loop_expectation_value, [i for i in range(self.M_sample)])
        else:
            results = []
            for ms in tqdm(range(self.M_sample), desc="Calculating expectation values"):
                results.append(self.loop_expectation_value(ms))
        return np.mean(results), np.std(results)
    
    def loop_expectation_value(self, ms):
        """
        Loop for computing expectation values on a single snapshot.

        Args:
            ms (int): Index of the snapshot.

        Returns:
            float: Weighted expectation value for the snapshot.
        """
        if isinstance(self.TE_PAI_Circuits[ms], str):
            results = np.array([self.GAMMA[ms]* simulator.get_expectation_value(self.deserialize_circuit( self.TE_PAI_Circuits[ms]), self.observable)])
        else:
             results = np.array([self.GAMMA[ms]* simulator.get_expectation_value(self.TE_PAI_Circuits[ms], self.observable)])
        return results
        
    def run_te_pai(self, observable, verbose=True):
        """
        Main entry point for executing TE-PAI.

        Args:
            observable (PauliSumOp or compatible): Observable to evaluate.
            verbose (bool): Show timing info during circuit generation.

        Returns:
            tuple: Mean and standard deviation of measured observable.
        """
        t_start = time.time()
        self.gen_te_pai_circuits(verbose)
        print("Time to generate TE_PAI circuits:", time.time() - t_start)
        return self.get_expectation_value(observable)

    def gen_cir_from_index(self, index):
        """
        Generate rotation gate instructions from sampled indices.

        Args:
            index (List[Tuple]): Indices corresponding to selected terms and angles.

        Returns:
            Tuple[List[Tuple], float]: List of gates and associated global prefactor GAMMA.
        """
        gates, sign = [], 1
        for i, inde in enumerate(index):
            for j, val in inde:
                pauli, ind, coef = self.terms[i][j]
                if val == 3:
                    sign *= -1
                    gates.append((pauli, ind, np.pi))
                else:
                    gates.append((pauli, ind, np.sign(coef) * self.delta))

        return gates, sign * self.gamma

    def rgate(self, pauli, r):
        """
        Construct a rotation gate for a specified Pauli interaction.

        Args:
            pauli (str): Pauli type ('X', 'Z', 'XX', 'YY', 'ZZ').
            r (float): Rotation angle.

        Returns:
            Gate: Corresponding Qiskit rotation gate.
        """
        return {
            "X": RXGate(r),
            "Z": RZGate(r),
            "XX": RXXGate(r),
            "YY": RYYGate(r),
            "ZZ": RZZGate(r),
        }[pauli]

    def gen_quantum_circuit(self, gates):
        if self.serialize:
            return self.gen_Qasm_circuit(gates)

        circ = QuantumCircuit(self.nq)
        for pauli, qubits, coef in gates:
            circ.append(self.rgate(pauli, coef), qubits)

        if hasattr(self, "_cached_init_circ") and self._cached_init_circ is not None:
            circ = self._cached_init_circ.compose(circ, front=True)

        base_circuit = transpile(circ, basis_gates=self.backend.configuration().basis_gates, optimization_level=3)
        clifford_gates_list = []
        transpiled_circ_list = []

        for _ in range(self.shadow_size):
            circuit_copy = QuantumCircuit(self.nq)
            circuit_copy.compose(base_circuit, inplace=True)
            clifford_labels = []
            for qubit in range(self.nq):
                label = self.random_clifford_gate()
                clifford_circ = self.precompiled_cliffords[label]
                circuit_copy.compose(clifford_circ, qubits=[qubit], inplace=True)
                clifford_labels.append(label)

            circuit_copy.measure_all()
            transpile_circ_for_backend=transpile(circuit_copy, backend=self.backend, optimization_level=0)
            transpiled_circ_list.append(transpile_circ_for_backend)
            clifford_gates_list.append(clifford_labels)

        depth = transpiled_circ_list[0].depth()
        return clifford_gates_list, transpiled_circ_list, depth

    def gen_Qasm_circuit(self, gates):
        """
        Generate a QASM representation of a quantum circuit with custom gate support.

        Args:
            gates (List[Tuple]): List of (pauli, qubit indices, angle) triples.

        Returns:
            str: OpenQASM 2.0 string for the circuit.
        """
        nq = self.nq
        from qiskit.qasm2 import dumps
        import re

        gate_definitions_list = """
                gate rxx(theta) a, b { h a; h b; cx a, b;rz(theta) b; cx a, b; h a; h b;}
                gate ryy(theta) a, b { ry(-pi/2) a; ry(-pi/2) b; cx a, b; rz(theta) b; cx a, b; ry(pi/2) a; ry(pi/2) b; }
                gate rzz(theta) a, b { cx a, b; rz(theta) b; cx a, b; }
                """
        qasm_circ = f"""OPENQASM 2.0; include "qelib1.inc";{gate_definitions_list}; qreg q[{nq}];"""
        # = dumps(circuit)  # Convert circuit to OpenQASM 2.0
        import re
        if isinstance(self.init_state, (np.ndarray, list)):
            circ = QuantumCircuit(nq)
            circ.initialize(self.init_state, [
                            i for i in range(nq)], normalize=True)
            circ = transpile(circ, basis_gates=["rx", "ry", "rz", "cx"])
            qasm_init = dumps(circ)
            qasm_circ = re.sub(
                r"(qreg q\[\d+\];\s*)", r"\1" +
                gate_definitions_list + "\n", qasm_init
            )
        elif isinstance(self.init_state, QuantumCircuit):
            circ = transpile(self.init_state, basis_gates=[
                             "rx", "ry", "rz", "cx"])
            qasm_init = dumps(circ)
            qasm_circ = re.sub(
                r"(qreg q\[\d+\];\s*)", r"\1" +
                gate_definitions_list + "\n", qasm_init
            )
        for pauli, qubits, coef in gates:
            gate = f"r" + str.lower(pauli)
            if len(qubits) == 2:
                qasm_circ += f"{gate}({coef}) q[{qubits[0]}],q[{qubits[1]}];"
            else:
                qasm_circ += f"{gate}({coef}) q[{qubits[0]}];"
        return qasm_circ

    def deserialize_circuit(self, qasm_str):
        """
        Deserialize an OpenQASM string into a Qiskit QuantumCircuit.

        Args:
            qasm_str (str): QASM string.

        Returns:
            QuantumCircuit: The deserialized circuit object.
        """
        from qiskit.qasm2 import loads, CustomInstruction

        rxx_custom = CustomInstruction(
            name="rxx", num_params=1, num_qubits=2, builtin=False, constructor=RXXGate
        )
        ryy_custom = CustomInstruction(
            name="ryy", num_params=1, num_qubits=2, builtin=False, constructor=RYYGate
        )
        rzz_custom = CustomInstruction(
            name="rzz", num_params=1, num_qubits=2, builtin=False, constructor=RZZGate)
        custom_instruction_list = [rxx_custom, ryy_custom, rzz_custom]
        return loads(qasm_str,  custom_instructions=custom_instruction_list)

    def get_average_depth(self):
        try : 
            if self.serialize: 
                res=0
                for circ in self.TE_PAI_Circuits:
                    qreg_match = re.search(r"qreg\s+(\w+)\[(\d+)\];", circ)
                    if not qreg_match:
                        raise ValueError("No qreg declaration found.")
                    prefix, size = qreg_match.group(1), int(qreg_match.group(2))
                    qubit_depths = [0] * size
                    max_depth = 0
                    # Only consider these gates
                    gate_pattern = re.compile(
                        r"(rx|rz|rxx|ryy|rzz)\s*\([^)]*\)\s+([a-zA-Z_]+\[\d+\](?:\s*,\s*[a-zA-Z_]+\[\d+\])?)\s*;",
                        re.IGNORECASE
                    )
                    for match in gate_pattern.finditer(circ):
                        gate = match.group(1).lower()
                        qubit_args = match.group(2).replace(" ", "")
                        qubits = [int(q.split('[')[1][:-1]) for q in qubit_args.split(",")]
                        current_layer = max(qubit_depths[q] for q in qubits) + 1
                        for q in qubits:
                            qubit_depths[q] = current_layer
                        max_depth = max(max_depth, current_layer)
                    res+=max_depth
                return res/self.M_sample
            else :
                return np.mean([self.TE_PAI_Circuits[i].depth() for i in range(self.M_sample)])
        
        except Exception as e: 
            print("error : ", e)
 
    def random_clifford_gate(self, idx: int = None) -> UnitaryGate:
        """Get a random clifford gate from the Clifford gate set"""
        if idx is None:
            return secrets.choice(self.Clifford_Gate_set)
        return self.Clifford_Gate_set[idx]
    
  
        