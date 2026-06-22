# Written by: Chusei Kiumi, updated by Hugo PAGES
# Date: 2024-01-05

# Standard library imports
from multiprocessing.pool import ThreadPool
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
from . import pai, sampling, simulator


@dataclass
class TE_PAI:
    """
    TE-PAI Circuit Generator for Time-Evolution Probabilistic Angle Interpolation (TE-PAI)
    of quantum systems under Hamiltonian dynamics.

    This class constructs randomized circuits approximating Hamiltonian evolution using
    a probabilistic interpolation method that achieves angle precision within a defined
    threshold. It allows efficient simulation via circuit generation, serialization, and
    estimation of observable expectation values.
    """

    def __init__(self, hamil, numQs, delta, T,
                 trotter_steps=1.e-4, PAI_error=0.05, N_trotter_max=8000, M_sample_max=500,
                 init_state=None, serialize: bool = False):
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
        self.M_sample=M_sample_max
        
        if self.M_sample > M_sample_max:
            self.M_sample = M_sample_max
        if self.M_sample < 60:
            self.M_sample = 60
            
        self.num_processes = min(30, int(mp.cpu_count() * 0.25))

        print("overhead:", self.overhead)
        print("M_sample:", self.M_sample)
        print("Trot :", self.N)

        self.init_state = init_state
        self.serialize = serialize
        from qiskit import transpile

        if isinstance(self.init_state, (np.ndarray, list)):
            init_circ = QuantumCircuit(self.nq)
            init_circ.initialize(self.init_state, list(range(self.nq)), normalize=True)
            self._cached_init_circ = transpile(init_circ, optimization_level=1)

        elif isinstance(self.init_state, QuantumCircuit):
            self._cached_init_circ = self.init_state.copy()






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
        t = time.time()
        index = sampling.batch_sampling(self.probs, self.M_sample)
        res = []
        chunk_size = max(1, self.M_sample// (self.num_processes*5))
        with mp.Pool(self.num_processes) as pool:
            res += pool.map(self.gen_cir_from_index, index, chunksize=chunk_size)
        print("time to generate from index :", time.time() - t)
        gates_array, self.GAMMA = zip(*res)
        self.GAMMA = np.array(self.GAMMA)

        self.TE_PAI_Circuits = []
        t2 = time.time()
        with mp.Pool(self.num_processes) as pool:
            self.TE_PAI_Circuits += pool.map(
                self.gen_quantum_circuit, gates_array, chunksize=chunk_size)

        print("time to generate Quantum_circuit:", time.time() - t2)

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
        """
        Generate a Qiskit circuit from a list of rotation gates.

        Args:
            gates (List[Tuple]): List of (pauli, qubit indices, angle) triples.

        Returns:
            QuantumCircuit: Constructed circuit (Qiskit object or QASM string).
        """
        if self.serialize:
            return self.gen_Qasm_circuit(gates)
        circ = QuantumCircuit(self.nq)
        for pauli, qubits, coef in gates:
            circ.append(self.rgate(pauli, coef), qubits)

        # Apply the initialization circuit before the rotations
        if hasattr(self, "_cached_init_circ") and self._cached_init_circ is not None:
            circ = self._cached_init_circ.compose(circ, front=True)

        return circ

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
 