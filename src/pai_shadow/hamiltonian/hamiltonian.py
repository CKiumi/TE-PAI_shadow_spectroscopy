# Written by: Chusei Kiumi, updated by Hugo PAGES
# Date: 2024-01-05

# Standard library imports
import re
import math, time
from dataclasses import dataclass
from itertools import product
from typing import Callable, List, Tuple, Union

# Third-party imports
import numpy as np
from scipy import integrate
import scipy.sparse as sp
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import RXXGate, RYYGate, RZZGate, RZGate, RXGate
from scipy.sparse.linalg import eigsh

@dataclass
class Hamiltonian:
    """
    Hamiltonian class for representing and manipulating time-dependent quantum Hamiltonians.

    This class allows the user to define a Hamiltonian as a list of terms, where each term consists of:
    - A string identifying the type of Pauli operator (e.g., 'X', 'Z', 'XX', etc.),
    - A list of qubit indices the operator acts on,
    - A time-dependent coefficient function Callable[t → float].

    Key Features:
    - Supports arithmetic operations: addition (`+`) and tensor product (`*`) of Hamiltonians.
    - Can return the list of terms at a specific time `t`, or just the evaluated coefficients.
    - Computes the time-integrated $L_1$ norm of the Hamiltonian coefficients.
    - Converts the Hamiltonian into a matrix form using Kronecker products of Pauli matrices.
    - Generates a Qiskit `QuantumCircuit` using Trotterization to simulate the time evolution under the Hamiltonian.
    - Can output a serialized QASM string representation of the circuit.
    - Provides eigenvalues and energy gaps for analyzing the Hamiltonian spectrum.
    """
    
    nqubits: int #number of qubits
    terms: List[Tuple[str, List[int], Callable[[float], float]]] #terms of the Hamiltonian
    circ : QuantumCircuit= None # Quantum circuit representation of the hamiltonian
    serialize : bool= False #if the quantum circuit is serialize or not 
    H: np.ndarray = None #Matrix representation of the Hamiltonian 
    
    
    
    def __post_init__(self):
        pass
        # print("The number of qubit:" + str(self.nqubits))
        # print("Number of terms in the Hamiltonian:" + str(len(self.terms)))


    def get_term(self, t)-> list[tuple]:
        """return the terms of the Hamiltonian at a time t  

        Args:
            t (float): time

        Returns:
           list: terms of the hamiltonian 
        """
        return [(term[0], term[1], term[2](t)) for term in self.terms]

    def coefs(self, t: float)->list[float]:
        """return the list of coefs for each terms of the Hamiltonian at a time t

        Args:
            t (float): time

        Returns:
            list : list of coefs at a time t
        """
        return [term[2](t) for term in self.terms]

    def l1_norm(self, T: float)-> float:
        """return the l1 norm

        Args:
            T (float): time
        Returns:
            float : l1 norm
        """
        def fn(t): return np.linalg.norm(self.coefs(t), 1)
        return integrate.quad(fn, 0, T, limit=100)[0]

    def __len__(self)-> int:
        """len(Hamil), return the number of terms in the hamiltonian

        Returns:
            int: number of terms in the hamiltonian
        """
        return len(self.terms)

    def __add__(self, other):
        """addition between hamiltonian : H1+H2,
        return a new Hamiltonian with concatenation terms of H1 and H2. The number of qubits stay the same.
        Args:
           other (Hamiltonian): Hamiltonian to add

        Returns:
            Hamiltonian: Hamiltonian object,  H1+ H2
        """
        terms = []
        for term in self.terms:
            terms.append(term)
        for term in other.terms:
            terms.append(term)
        return Hamiltonian(self.nqubits, terms)

    def __mul__(self, other):
        """Multiplication of two Hamiltonian : H1*H2
        return a new hamiltonian such that : nq=nq1+nq2
        H(0<nq<nq1-1)= H1
        H(nq1<nq<nq2+nq1-1)= H2

        Args:
            other (Hamiltonian): hamiltonian to multiply 

        Returns:
            Hamiltonian object
        """
        terms = []
        for term in self.terms:
            terms.append(term)
        for term in other.terms:
            gate, qubits, coef = term[0], term[1], term[2]
            new_term = (gate, [i+self.nqubits for i in qubits], coef)
            terms.append(new_term)
        return Hamiltonian(self.nqubits+other.nqubits, terms)

    def __str__(self)->str:
        try:
            return str(self.name)
        except:
            return f"Hamiltonian object : nq={self.nqubits}, number of terms :{len(self.terms)}"


    def matrix(self, t=0)->np.ndarray:
        """Return the matrix representation of the hamiltonian 

        Returns:
            np.ndarray: Matrix representation of the Hamiltonian with size: (2**nq, 2**nq)
        """
        from functools import reduce
        dim=2**self.nqubits
        I = sp.identity(2, format='csr')
        X = sp.csr_matrix([[0, 1], [1, 0]])
        Y = sp.csr_matrix([[0, -1j], [1j, 0]])
        Z = sp.csr_matrix([[1, 0], [0, -1]])
        H = sp.csr_matrix((dim, dim), dtype=complex)
        
        for term in self.terms:
            tmp = [I]*self.nqubits
            if term[0] == 'Z':
                tmp[term[1][0]] = Z
            elif term[0] == 'X':
                tmp[term[1][0]] = X
            elif term[0] == 'Y':
                tmp[term[1][0]] = Y
            elif term[0] == 'XX':
                tmp[term[1][0]] = X
                tmp[term[1][1]] = X
            elif term[0] == 'YY':
                tmp[term[1][0]] = Y
                tmp[term[1][1]] = Y
            if term[0] == 'ZZ':
                tmp[term[1][0]] = Z
                tmp[term[1][1]] = Z
            H+=(term[2](t)*reduce(lambda A, B: sp.kron(A, B, format='csr'), tmp))
        return H

    def eigenvalues(self) -> np.ndarray:
        """Return the real part of the eigenvalues of a matrix , i.e the eigenenergies of the Hamiltonian from is matrix representation
        Args:
            matrix (np.ndarray): a square matrix that represant an Hamiltonian 

        Returns:
        np.ndarray: the eigenenergies of the Hamiltonian
        """
        matrix = self.matrix().toarray()
        eigval_list = np.linalg.eig(matrix)[0]
        eigval_list[np.abs(eigval_list) < 1.e-11] = 0
        return eigval_list.real
    
    def get_ground_and_excited_state(self,n=1,t=0, threshold: float = 1e-2, get_dict: bool = False) -> Tuple[float, float, np.ndarray, np.ndarray]:
        H = self.matrix(t)
        vals, vecs = eigsh(H, k=n+1, which='SA')
        ground_energy = vals[0]
        first_excited_energy = vals[n]
        ground_state = vecs[:, 0]
        excited_state = vecs[:, n]
        return ground_energy, first_excited_energy, ground_state, excited_state
    
    def energy_gap(self)->np.ndarray:
        """Calculate the energy gap between different energy levels.
        Remove duplicate energy levels.

        Returns:
            numpy.ndarray: Sorted array of unique energy gaps
        """
        rnd = 4
        energies = self.eigenvalues()
        energies[np.abs(energies) < 1.e-11] = 0
        unique_energies = np.unique(np.round(energies, rnd))
        E1, E2 = np.meshgrid(unique_energies, unique_energies)
        gaps = np.abs(E1 - E2)
        gaps = gaps[np.triu_indices_from(gaps, k=1)]
        unique_gaps = np.unique(np.round(gaps, rnd))
        unique_gaps[np.abs(unique_gaps) < 1.e-11] = 0
        return np.sort(unique_gaps)

###########################################################################################################################
##################################################### Quantum circuit #####################################################
###########################################################################################################################



    def gen_quantum_circuit(self, T: float, init_state: Union[np.ndarray, list, QuantumCircuit] = None, N_Trotter_steps: int = 1000, serialize=False, decompose : bool=False) -> QuantumCircuit:
        """Generate a qiskit quantum circuite given a list of gates.
        Args:
            gates (tuple[str, list[int], float]): The list of gates to generate the circuit from, in the form of : ("Gate", [nq1, nq2], parameters)
            exemple: ("XX",[2,3], 1) gate XX on qubit 2, 3 with parameter 1
            nq (int): total number of qubit
            init_state (QuantumCircuit, optional): A Quantum circuit to put at the beginning of the circuit. Defaults to None.

        Returns:
            QuantumCircuit: The quantum circuit representation of the given gates
        """
        if serialize:
            return self.gen_Qasm_circuit(T, init_state, N_Trotter_steps)
        nq = self.nqubits
        circ = QuantumCircuit(nq)
        steps = np.linspace(0, T, N_Trotter_steps)
        if isinstance(init_state, (np.ndarray, list)):
            circ.initialize(init_state, [i for i in range(nq)], normalize=True)
            if serialize:
                circ = transpile(circ, basis_gates=['rx', 'ry', 'rz', 'cx'])
        for step in steps:
            for pauli, qubits, coef in self.get_term(step):
                circ.append(self.__rgate(
                    pauli, 2*coef*T/N_Trotter_steps), qubits)

        if isinstance(init_state, QuantumCircuit):
            circ = init_state.compose(circ, [i for i in range(nq)])
        if decompose :
            circ=circ.decompose(["rzz"])
        self.circ=circ
        self.serialize=serialize
        return circ

    def __rgate(self, pauli:str, r: float):
        """list of rotation gate: used to go from the tuple representation 
        to the qiskit representation of the hamiltonian 

        Args:
            pauli (str): name Pauli matrix 
            r (float): rotation angle 

        Returns:
            qiskit rotation gate : qiskit rotation gate with angle r. 
        """
        return {
            "X": RXGate(r),
            "Z": RZGate(r),
            "XX": RXXGate(r),
            "YY": RYYGate(r),
            "ZZ": RZZGate(r),
        }[pauli]

    def gen_Qasm_circuit(self,  T: float, init_state: Union[np.ndarray, list, QuantumCircuit] = None, N_Trotter_steps: int = 1000)->str:
        """
        Generate a QASM quantum circuite given a list of gates.
        Args:
            T (float): time 
            init_state (Union[np.ndarray, list, QuantumCircuit]) : initial state of the qubits
            N_Trotter_steps (int) : number of trotter steps 
        Returns:
        str: Qasm representation of the hamiltonian
        
        """
        nq = self.nqubits
        from qiskit.qasm2 import dumps
        import re
        steps = np.linspace(0, T, N_Trotter_steps, endpoint=True)
        gate_definitions_list = """
            gate rxx(theta) a, b { h a; h b; cx a, b;rz(theta) b; cx a, b; h a; h b;}
            gate ryy(theta) a, b { ry(-pi/2) a; ry(-pi/2) b; cx a, b; rz(theta) b; cx a, b; ry(pi/2) a; ry(pi/2) b; }
            gate rzz(theta) a, b { cx a, b; rz(theta) b; cx a, b; }
            """
        qasm_circ = f"""OPENQASM 2.0; include "qelib1.inc";{gate_definitions_list}; qreg q[{nq}];"""
        # = dumps(circuit)  # Convert circuit to OpenQASM 2.0
        import re
        if isinstance(init_state, (np.ndarray, list)):
            circ = QuantumCircuit(nq)
            circ.initialize(init_state, [i for i in range(nq)], normalize=True)
            circ = transpile(circ, basis_gates=['rx', 'ry', 'rz', 'cx'])
            qasm_init = dumps(circ)
            qasm_circ = re.sub(
                r"(qreg q\[\d+\];\s*)", r"\1" + gate_definitions_list + "\n", qasm_init)
        elif isinstance(init_state, QuantumCircuit):
            circ = transpile(init_state, basis_gates=['rx', 'ry', 'rz', 'cx'])
            qasm_init = dumps(circ)
            qasm_circ = re.sub(
                r"(qreg q\[\d+\];\s*)", r"\1" + gate_definitions_list + "\n", qasm_init)
        for step in steps:
            for pauli, qubits, coef in self.get_term(step):
                gate = f"r"+str.lower(pauli)
                if len(qubits) == 2:
                    qasm_circ += f"{gate}({2*coef*T/N_Trotter_steps}) q[{qubits[0]}],q[{qubits[1]}];"
                elif len(qubits) == 1:
                    qasm_circ += f"{gate}({2*coef*T/N_Trotter_steps}) q[{qubits[0]}];"
                else:
                    print(
                        "gen_Qasm_circuit: ERROR only working with 1 and 2 qubits gates")
        return qasm_circ


    def get_depth(self):
        try : 
            if isinstance(self.circ, str):
                qreg_match = re.search(r"qreg\s+(\w+)\[(\d+)\];", self.circ)
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

                for match in gate_pattern.finditer(self.circ):
                    gate = match.group(1).lower()
                    qubit_args = match.group(2).replace(" ", "")
                    qubits = [int(q.split('[')[1][:-1]) for q in qubit_args.split(",")]

                    current_layer = max(qubit_depths[q] for q in qubits) + 1

                    for q in qubits:
                        qubit_depths[q] = current_layer

                    max_depth = max(max_depth, current_layer)
            if isinstance(self.circ, QuantumCircuit):
                return self.circ.depth()
        except Exception as e: 
            print("error : ", e)
 