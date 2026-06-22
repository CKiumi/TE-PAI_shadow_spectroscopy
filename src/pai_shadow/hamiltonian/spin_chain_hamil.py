# Written by: Hugo PAGES
# Date: 2024-01-05

# Standard library imports
from dataclasses import dataclass
from itertools import product
from typing import Callable, List, Tuple

# Third-party imports
import numpy as np
from scipy import integrate

# Local application imports
from .hamiltonian import Hamiltonian


@dataclass
class Spin_Chain_Hamil(Hamiltonian):
    """
    Spin_Chain_Hamil defines a time-dependent spin chain Hamiltonian with nearest-neighbor 
    interactions and local Z fields.

    The Hamiltonian takes the form:
        H(t) = J(t) ∑ (XXₖ,ₖ₊₁ + YYₖ,ₖ₊₁ + ZZₖ,ₖ₊₁) + ∑ fₖ Zₖ

    Parameters:
    - n (int): Number of qubits (spins) in the chain.
    - freqs (list[float]): Local Z field strengths for each qubit.
    - func (callable, optional): A time-dependent function J(t) for coupling strength.
    If not provided, defaults to J(t) = cos(20πt), representing periodic modulation.

    Periodic boundary conditions are applied, connecting the last qubit to the first.

    This class is built on top of the generic Hamiltonian class and uses time-dependent coefficients
    to support simulation of dynamic spin chains.
    """
    nqubits: int
    terms: List[Tuple[str, List[int], Callable[[float], float]]]

    def __init__(self, n, freqs, func=None):
        if func is not None:
            def J(t):
                return func(t)
        else:
            def J(_):
                return 1

        terms = [
            (gate, [k, (k + 1)], J)
            for k, gate in product(range(n-1), ["XX", "YY", "ZZ"])
        ]
        terms += [("Z", [k], lambda t, k=k: freqs[k]) for k in range(n)]
        self.name = f"SpinChain_nq{n}"
        super().__init__(n, terms)
