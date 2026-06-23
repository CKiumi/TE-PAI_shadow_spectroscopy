# Backend-independent Pauli-sum Hamiltonians.
#
# A Hamiltonian is represented purely as a weighted sum of Pauli strings,
#     H(t) = sum_j  c_j(t) * P_j ,
# where each P_j is a tensor product of single-qubit Pauli operators acting on
# a chosen subset of qubits and c_j is a (possibly time-dependent) coefficient.
#
# This module depends only on numpy/scipy. It knows nothing about quantum
# circuits or any simulation backend: turning a Hamiltonian into a circuit
# (Trotterization, ...) is the responsibility of a separate layer.

from __future__ import annotations

from functools import reduce
from typing import Callable, List, Sequence, Tuple, Union

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

# A coefficient is either a constant number or a function of time.
Coefficient = Union[float, complex, Callable[[float], Union[float, complex]]]
# A term is (pauli_string, qubits, coefficient), e.g. ("XX", [0, 1], 1.0).
Term = Tuple[str, Sequence[int], Coefficient]

# Single-qubit Pauli matrices (sparse).
_PAULI_1Q = {
    "I": sp.csr_matrix(np.eye(2, dtype=complex)),
    "X": sp.csr_matrix(np.array([[0, 1], [1, 0]], dtype=complex)),
    "Y": sp.csr_matrix(np.array([[0, -1j], [1j, 0]], dtype=complex)),
    "Z": sp.csr_matrix(np.array([[1, 0], [0, -1]], dtype=complex)),
}

# Threshold above which the ground/excited search uses sparse iterative
# diagonalisation instead of a dense eigendecomposition.
_DENSE_DIM_LIMIT = 1 << 12  # 4096 == 12 qubits


def _eval(coeff: Coefficient, t: float) -> complex:
    """Evaluate a coefficient at time ``t`` (constants pass through)."""
    return coeff(t) if callable(coeff) else coeff


class Hamiltonian:
    """A weighted sum of Pauli strings with optional time-dependent coefficients.

    Parameters
    ----------
    num_qubits:
        Number of qubits the Hamiltonian acts on.
    terms:
        Iterable of ``(pauli, qubits, coeff)``:

        - ``pauli``  -- string of single-qubit Pauli labels, one per entry of
          ``qubits`` (e.g. ``"X"``, ``"ZZ"``, ``"XYZ"``). Allowed chars: I X Y Z.
        - ``qubits`` -- qubit indices the Pauli string acts on (same length as
          ``pauli``).
        - ``coeff``  -- a real/complex number, or a callable ``t -> number``.
    name:
        Optional human-readable label.

    Notes
    -----
    Qubit 0 is the least-significant tensor factor (little-endian), matching the
    simulation backends, so eigenvectors can be used directly as circuit initial
    states. The Hamiltonian is assumed Hermitian for the spectral helpers.
    """

    def __init__(self, num_qubits: int, terms: List[Term], name: str | None = None):
        self.nqubits = int(num_qubits)
        self.terms: List[Term] = [self._validate_term(term) for term in terms]
        self.name = name

    # ------------------------------------------------------------------ #
    #  Validation / basic accessors                                       #
    # ------------------------------------------------------------------ #
    def _validate_term(self, term: Term) -> Term:
        pauli, qubits, coeff = term
        qubits = list(qubits)
        if len(pauli) != len(qubits):
            raise ValueError(
                f"Pauli string {pauli!r} and qubits {qubits} differ in length."
            )
        for p in pauli:
            if p not in _PAULI_1Q:
                raise ValueError(f"Unknown Pauli label {p!r} in term {pauli!r}.")
        for q in qubits:
            if not 0 <= q < self.nqubits:
                raise ValueError(
                    f"Qubit index {q} out of range [0, {self.nqubits}) in term {pauli!r}."
                )
        if len(set(qubits)) != len(qubits):
            raise ValueError(f"Repeated qubit index in term {pauli!r} on {qubits}.")
        return (pauli, qubits, coeff)

    @property
    def num_qubits(self) -> int:
        """Alias for :attr:`nqubits`."""
        return self.nqubits

    def __len__(self) -> int:
        return len(self.terms)

    def get_term(self, t: float = 0.0) -> List[Tuple[str, List[int], complex]]:
        """Return the terms with coefficients evaluated at time ``t``."""
        return [(p, list(q), _eval(c, t)) for p, q, c in self.terms]

    def coefs(self, t: float = 0.0) -> List[complex]:
        """Return the coefficient of every term evaluated at time ``t``."""
        return [_eval(c, t) for _, _, c in self.terms]

    def l1_norm(self, T: float) -> float:
        """Time-integrated L1 norm of the coefficients on ``[0, T]``."""
        from scipy import integrate

        def fn(t):
            return float(np.linalg.norm(np.real(self.coefs(t)), 1))

        return integrate.quad(fn, 0, T, limit=100)[0]

    # ------------------------------------------------------------------ #
    #  Algebra                                                            #
    # ------------------------------------------------------------------ #
    def __add__(self, other: "Hamiltonian") -> "Hamiltonian":
        """Sum of two Hamiltonians on the same qubits (term concatenation)."""
        if self.nqubits != other.nqubits:
            raise ValueError(
                f"Cannot add Hamiltonians on {self.nqubits} and {other.nqubits} qubits."
            )
        return Hamiltonian(self.nqubits, [*self.terms, *other.terms])

    def __mul__(self, other: "Hamiltonian") -> "Hamiltonian":
        """Tensor product: ``other`` is appended on fresh qubits above ``self``."""
        shifted = [
            (p, [q + self.nqubits for q in qs], c) for p, qs, c in other.terms
        ]
        return Hamiltonian(self.nqubits + other.nqubits, [*self.terms, *shifted])

    def __str__(self) -> str:
        if self.name:
            return str(self.name)
        return f"Hamiltonian(nqubits={self.nqubits}, terms={len(self.terms)})"

    __repr__ = __str__

    # ------------------------------------------------------------------ #
    #  Matrix representation & spectrum (numpy/scipy only)                #
    # ------------------------------------------------------------------ #
    def matrix(self, t: float = 0.0) -> sp.csr_matrix:
        """Sparse matrix representation of ``H(t)`` (shape ``2**nq x 2**nq``)."""
        dim = 1 << self.nqubits
        H = sp.csr_matrix((dim, dim), dtype=complex)
        for pauli, qubits, coeff in self.terms:
            factors = [_PAULI_1Q["I"]] * self.nqubits
            for p, q in zip(pauli, qubits):
                factors[q] = _PAULI_1Q[p]
            # little-endian: qubit 0 is the least-significant factor (right-most
            # in the Kronecker product), matching the simulation backends.
            term_op = reduce(lambda A, B: sp.kron(A, B, format="csr"), factors[::-1])
            H = H + _eval(coeff, t) * term_op
        return H

    def to_dense(self, t: float = 0.0) -> np.ndarray:
        """Dense matrix representation of ``H(t)``."""
        return self.matrix(t).toarray()

    def eigh(self, t: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """Full Hermitian eigendecomposition ``(eigenvalues, eigenvectors)``.

        Eigenvalues are real and returned in ascending order; column ``i`` of
        the eigenvector array is the eigenvector for ``eigenvalues[i]``.
        """
        return np.linalg.eigh(self.to_dense(t))

    def eigenvalues(self, t: float = 0.0, tol: float = 1e-11) -> np.ndarray:
        """Sorted real eigenenergies of ``H(t)`` (tiny values snapped to 0)."""
        vals = np.linalg.eigvalsh(self.to_dense(t))
        vals[np.abs(vals) < tol] = 0.0
        return vals

    def get_ground_and_excited_state(
        self, n: int = 1, t: float = 0.0
    ) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """Return ``(E_0, E_n, |E_0>, |E_n>)`` at time ``t``.

        Uses a dense decomposition for small systems and sparse iterative
        diagonalisation (``eigsh``) for larger ones.
        """
        dim = 1 << self.nqubits
        if not 0 <= n < dim:
            raise ValueError(f"Excited index n={n} out of range [0, {dim}).")
        if dim <= _DENSE_DIM_LIMIT:
            vals, vecs = self.eigh(t)
        else:
            k = min(n + 1, dim - 1)
            vals, vecs = eigsh(self.matrix(t), k=k, which="SA")
            order = np.argsort(vals)
            vals, vecs = vals[order], vecs[:, order]
        return float(vals[0]), float(vals[n]), vecs[:, 0], vecs[:, n]

    def energy_gap(self, t: float = 0.0, rnd: int = 4, tol: float = 1e-11) -> np.ndarray:
        """Sorted array of the unique energy gaps |E_a - E_b| at time ``t``."""
        energies = self.eigenvalues(t, tol=tol)
        unique_energies = np.unique(np.round(energies, rnd))
        E1, E2 = np.meshgrid(unique_energies, unique_energies)
        gaps = np.abs(E1 - E2)[np.triu_indices(len(unique_energies), k=1)]
        unique_gaps = np.unique(np.round(gaps, rnd))
        unique_gaps[np.abs(unique_gaps) < tol] = 0.0
        return np.sort(unique_gaps)


class Heisenberg_Hamil(Hamiltonian):
    """Time-independent Heisenberg model on ``n`` qubits (paper Fig. 1/2 model).

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


class Ising_Hamil(Hamiltonian):
    """Transverse/longitudinal-field Ising model on ``n`` qubits (paper Fig. 3 model).

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


__all__ = ["Hamiltonian", "Heisenberg_Hamil", "Ising_Hamil"]
