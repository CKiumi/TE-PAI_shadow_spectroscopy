# Heisenberg spin-chain Hamiltonian (backend-independent).

from __future__ import annotations

from .hamiltonian import Hamiltonian


class Heisenberg_Hamil(Hamiltonian):
    """Time-independent Heisenberg model on ``n`` qubits.

    H = jx * sum XX + jy * sum YY + jz * sum ZZ over the chosen geometry.

    Parameters
    ----------
    n:
        Number of qubits.
    jx, jy, jz:
        Coupling constants of the XX, YY and ZZ interactions.
    periodic:
        If True use periodic boundary conditions (qubit ``n-1`` couples to 0);
        otherwise open boundary conditions. Ignored when ``fully_connected``.
    fully_connected:
        If True every pair of qubits interacts (all-to-all).
    """

    def __init__(
        self,
        n: int,
        jx: float,
        jy: float,
        jz: float,
        periodic: bool = False,
        fully_connected: bool = False,
    ):
        self.jx, self.jy, self.jz = jx, jy, jz
        self.periodic = periodic
        self.fully_connected = fully_connected
        couplings = {"XX": jx, "YY": jy, "ZZ": jz}

        if fully_connected:
            pairs = [[k, j] for k in range(n) for j in range(k + 1, n)]
        elif periodic:
            pairs = [[k, (k + 1) % n] for k in range(n)]
        else:
            pairs = [[k, k + 1] for k in range(n - 1)]

        terms = [
            (gate, list(pair), coef)
            for pair in pairs
            for gate, coef in couplings.items()
        ]

        name = f"Heisenberg_Jx{jx}_Jy{jy}_Jz{jz}_nq{n}"
        if fully_connected:
            name += "_fully_connected"
        elif periodic:
            name += "_periodic"
        super().__init__(n, terms, name=name)
