# Transverse/longitudinal-field Ising Hamiltonian (backend-independent).

from __future__ import annotations

from .hamiltonian import Hamiltonian


class Ising_Hamil(Hamiltonian):
    """Time-independent Ising model on ``n`` qubits.

    H = -J * sum ZZ  - g * sum X  - h * sum Z

    Parameters
    ----------
    n:
        Number of qubits.
    J:
        ZZ coupling strength.
    transverse:
        Transverse field strength ``g`` (X terms). ``None`` disables it.
    longitudinal:
        Longitudinal field strength ``h`` (Z terms). ``None`` disables it.
    periodic:
        Periodic boundary conditions for the ZZ chain. Ignored when
        ``fully_connected``.
    fully_connected:
        All-to-all ZZ interactions.
    """

    def __init__(
        self,
        n: int,
        J: float,
        transverse: float | None = None,
        longitudinal: float | None = None,
        periodic: bool = True,
        fully_connected: bool = False,
    ):
        self.J = J
        self.g = float(transverse) if isinstance(transverse, (int, float)) else 0.0
        self.h = float(longitudinal) if isinstance(longitudinal, (int, float)) else 0.0
        self.periodic = periodic
        self.fully_connected = fully_connected

        if fully_connected:
            pairs = [[k, j] for k in range(n) for j in range(k + 1, n)]
        elif periodic:
            pairs = [[k, (k + 1) % n] for k in range(n)]
        else:
            pairs = [[k, k + 1] for k in range(n - 1)]

        terms = [("ZZ", list(pair), -J) for pair in pairs]
        if self.g:
            terms += [("X", [k], -self.g) for k in range(n)]
        if self.h:
            terms += [("Z", [k], -self.h) for k in range(n)]

        name = f"Ising_J{J}_h{self.h}_g{self.g}_nq{n}"
        if fully_connected:
            name += "_fully_connected"
        elif periodic:
            name += "_periodic"
        super().__init__(n, terms, name=name)
