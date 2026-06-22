# Written by: Hugo PAGES
# Date: 2024-01-05

# Standard library imports
from dataclasses import dataclass
from typing import Callable, List, Tuple

# Third-party imports
import numpy as np
from qiskit import QuantumCircuit
from scipy.sparse.linalg import eigsh

# Local application imports
from .hamiltonian import Hamiltonian


@dataclass
class Ising_Hamil(Hamiltonian):
    """
    Ising_Hamil is a subclass of Hamiltonian that defines a time-independent Ising model 
    Hamiltonian with optional transverse and longitudinal fields.

    The Hamiltonian has the form:
        H = -J ∑ ZZₖ,ₖ₊₁ - g ∑ Xₖ - h ∑ Zₖ

    Parameters:
    - n (int): Number of qubits (spins).
    - J (float): Coupling strength between neighboring spins (ZZ interaction).
    - transverse (float, optional): Transverse field strength (X terms). Default is None.
    - longitudinal (float, optional): Longitudinal field strength (Z terms). Default is None.

    Periodic boundary conditions are assumed (i.e., qubit `n` couples to qubit `0`).

    The class sets all coefficients as time-independent functions to stay compatible 
    with the base Hamiltonian class.
    """
    nqubits: int
    terms: List[Tuple[str, List[int], Callable[[float], float]]]


    def __init__(self, n, J, transverse: float = None, longitudinal: float = None, fully_connected : bool= False, bundary_conditions: bool = True):
            self.fully_connected=fully_connected
            if fully_connected:
                terms = [("ZZ", [k, j], lambda t: -1 * J) for k in range(n) for j in range(k + 1, n)]
            else : 
                if bundary_conditions:
                    terms = [("ZZ", [k, (k + 1) % n], lambda t: -1 * J) for k in range(n)]
                else:
                    terms = [("ZZ", [k, (k + 1)], lambda t: -1 * J) for k in range(n - 1)]
            self.J = J
            self.g = 0
            self.h = 0
            if isinstance(transverse, (float, int)):
                self.g = transverse
                terms += [("X", [k], lambda t: -1*self.g) for k in range(n)]
            if isinstance(longitudinal, (float, int)):
                self.h = longitudinal
                terms += [("Z", [k], lambda t: -1*self.h) for k in range(n)]
            self.name = f"Ising_J{J}_h{self.h}_g{self.g}_nq{n}"
            if fully_connected:
                self.name += "_fully_connected"
            elif bundary_conditions:
                self.name += "_periodic_boundary_conditions"
            print(f" instance of Hamiltonian : {self.name} created")
            super().__init__(n, terms)



    
    
    

    def ground_state(self,  n_trotter = 20, dt = 0.1 ):
        """ Return the ground_state of the Ising Hamiltonian 
        """
        Ground_state = QuantumCircuit(self.nqubits)
        Ground_state.h([i for i in range(self.nqubits)])
        for n in range(n_trotter):
            for q in range(self.nqubits - 1):
                Ground_state.rzz(2 * self.J * dt, q,q+1)  
            Ground_state.rzz(2 * self.J * dt, 0, self.nqubits-1)  
            for q in range(self.nqubits):
                Ground_state.rx(2 * self.g * dt, q)
        return Ground_state

    def Excited_state(self,  n_trotter = 20, dt = 0.1 ):
        """ Return the ground_state of the Ising Hamiltonian 
            If Transverse: 
                |++...+>

        Returns:
            _type_: _description_
        """
        Excited_state= QuantumCircuit(self.nqubits)
        Excited_state.x(0) 
        Excited_state.h(0) 
        Excited_state.h([i for i in range(1,self.nqubits)])
        for n in range(n_trotter):
            for q in range(self.nqubits - 1):
                Excited_state.rzz(2 * self.J * dt, q,q+1)  
            Excited_state.rzz(2 * self.J * dt, 0, self.nqubits -1)  
            for q in range(self.nqubits):
                 Excited_state.rx(2 * self.g * dt, q)
        return  Excited_state


    def superposition(self,n_trotter = 20, dt = 0.1 ):
        n_qubits = self.nqubits+ 1 # total qubits (include one ancilla)
        superposition = QuantumCircuit(n_qubits, n_qubits)
        superposition.h(0)
        superposition.cx(0, 1)
        for q in range(1, n_qubits):
            superposition.h(q)
        for n in range(n_trotter):
            for q in range(1,n_qubits - 1):
                superposition.rzz(2 * self.J * dt, q,q+1)  
            superposition.rzz(2 * self.J * dt, 1, n_qubits -1)  
            for q in range(1, n_qubits):
                 superposition.rx(2 * self.g * dt, q)
        superposition.h(0)
        superposition.measure(0,0)
        return superposition

    def get_trotter_steps_from_depth(self, depth: int) -> int:
        """
        Compute the number of Trotter steps based on the desired circuit depth.

        Args:
            depth (int): Desired circuit depth.

        Returns:
            int: Number of Trotter steps.
        """
        if self.fully_connected : 
            N_trot=int((depth-(self.nqubits-2.4)) / (1.1*self.nqubits + 1.4))
        else:
            N_trot=int(depth/ (self.nqubits + 2))
        return N_trot