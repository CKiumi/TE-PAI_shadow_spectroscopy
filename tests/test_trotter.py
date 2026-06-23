"""Tests for deterministic Trotter time-evolution circuits."""

import numpy as np
import pytest
from scipy.linalg import expm

from pai_shadow.backend import get_backend
from pai_shadow.hamil import Hamiltonian, Heisenberg_Hamil, Ising_Hamil
from pai_shadow.trotter import trotter_circuit

_P = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def H_little_endian(hamil, t=0.0):
    """Dense H(t) with qubit 0 as the least-significant bit (backend convention)."""
    n = hamil.nqubits
    H = np.zeros((1 << n, 1 << n), dtype=complex)
    for pauli, qubits, coef in hamil.get_term(t):
        mats = [_P["I"]] * n
        for p, q in zip(pauli, qubits):
            mats[q] = _P[p]
        op = mats[n - 1]
        for q in range(n - 2, -1, -1):
            op = np.kron(op, mats[q])
        H += coef * op
    return H


def random_state(n, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(1 << n) + 1j * rng.standard_normal(1 << n)
    return v / np.linalg.norm(v)


def fidelity(a, b):
    return abs(np.vdot(a, b)) ** 2


def exact_state(hamil, t, psi0):
    return expm(-1j * H_little_endian(hamil) * t) @ psi0


# --------------------------------------------------------------------------- #
def test_single_term_is_exact():
    # One commuting term -> first-order Trotter is exact for any n_steps.
    h = Hamiltonian(1, [("Z", [0], 0.7)])
    psi0 = random_state(1, seed=1)
    c = trotter_circuit(h, t=1.3, n_steps=1, init_state=psi0)
    sv = get_backend("qulacs").statevector(c)
    assert np.isclose(fidelity(sv, exact_state(h, 1.3, psi0)), 1.0, atol=1e-10)


def test_zero_time_is_identity():
    h = Heisenberg_Hamil(2, 1, 1, 1)
    psi0 = random_state(2, seed=2)
    c = trotter_circuit(h, t=0.0, n_steps=10, init_state=psi0)
    assert len(c) == 0
    sv = get_backend("qulacs").statevector(c)
    assert np.isclose(fidelity(sv, psi0), 1.0, atol=1e-12)


def test_converges_to_exact():
    h = Ising_Hamil(3, J=1.0, transverse=1.0, periodic=False)  # non-commuting terms
    psi0 = random_state(3, seed=3)
    t = 1.0
    c = trotter_circuit(h, t, n_steps=400, init_state=psi0)
    sv = get_backend("qulacs").statevector(c)
    assert fidelity(sv, exact_state(h, t, psi0)) > 0.999


def test_gate_counts_and_structure():
    h = Heisenberg_Hamil(4, 1, 1, 1)            # 3*(4-1) = 9 terms
    n_steps = 5
    c1 = trotter_circuit(h, t=1.0, n_steps=n_steps)
    assert len(c1) == n_steps * len(h)          # first order


def test_init_state_passthrough():
    h = Heisenberg_Hamil(2, 1, 1, 1)
    psi0 = random_state(2, seed=5)
    c = trotter_circuit(h, t=0.5, n_steps=3, init_state=psi0)
    assert c.init_state is psi0


def test_invalid_arguments():
    h = Heisenberg_Hamil(2, 1, 1, 1)
    with pytest.raises(ValueError):
        trotter_circuit(h, t=1.0, n_steps=0)
