"""Tests for the backend-independent Hamiltonian type and its subclasses."""

import numpy as np
import pytest

from pai_shadow.hamiltonian import (
    Hamiltonian,
    Heisenberg_Hamil,
    Ising_Hamil,
)

# Single-qubit Pauli matrices for reference comparisons.
I2 = np.eye(2)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


# --------------------------------------------------------------------------- #
#  Construction & validation                                                  #
# --------------------------------------------------------------------------- #
def test_basic_attributes():
    h = Hamiltonian(3, [("X", [0], 1.0), ("ZZ", [1, 2], -0.5)])
    assert h.nqubits == 3
    assert h.num_qubits == 3
    assert len(h) == 2


@pytest.mark.parametrize(
    "term",
    [
        ("XX", [0], 1.0),       # pauli/qubits length mismatch
        ("XW", [0, 1], 1.0),    # unknown Pauli label
        ("X", [5], 1.0),        # qubit out of range
        ("ZZ", [1, 1], 1.0),    # repeated qubit
    ],
)
def test_invalid_terms_raise(term):
    with pytest.raises(ValueError):
        Hamiltonian(2, [term])


# --------------------------------------------------------------------------- #
#  Coefficients (constant and time-dependent)                                 #
# --------------------------------------------------------------------------- #
def test_constant_and_time_dependent_coefs():
    h = Hamiltonian(1, [("Z", [0], 2.0), ("X", [0], lambda t: np.cos(t))])
    assert h.coefs(0.0) == [2.0, 1.0]
    assert np.isclose(h.coefs(np.pi)[1], -1.0)
    terms_t = h.get_term(0.0)
    assert terms_t[0] == ("Z", [0], 2.0)


def test_l1_norm_constant():
    h = Hamiltonian(1, [("Z", [0], 2.0), ("X", [0], -3.0)])
    # integral over [0, T] of (|2| + |-3|) dt = 5 * T
    assert np.isclose(h.l1_norm(4.0), 20.0)


# --------------------------------------------------------------------------- #
#  Matrix representation & qubit-ordering convention                          #
# --------------------------------------------------------------------------- #
def test_single_qubit_matrix():
    assert np.allclose(Hamiltonian(1, [("Z", [0], 1.0)]).to_dense(), Z)
    assert np.allclose(Hamiltonian(1, [("X", [0], 1.0)]).to_dense(), X)


def test_qubit_ordering_convention():
    # Qubit 0 is the most significant tensor factor (left-most in the kron).
    assert np.allclose(Hamiltonian(2, [("Z", [0], 1.0)]).to_dense(), np.kron(Z, I2))
    assert np.allclose(Hamiltonian(2, [("Z", [1], 1.0)]).to_dense(), np.kron(I2, Z))


def test_two_qubit_term_matrix():
    expected = np.kron(X, X)
    assert np.allclose(Hamiltonian(2, [("XX", [0, 1], 1.0)]).to_dense(), expected)


# --------------------------------------------------------------------------- #
#  Spectrum                                                                   #
# --------------------------------------------------------------------------- #
def test_eigenvalues_single_z():
    vals = Hamiltonian(1, [("Z", [0], 1.0)]).eigenvalues()
    assert np.allclose(np.sort(vals), [-1.0, 1.0])


def test_heisenberg_two_qubit_spectrum():
    # H = XX + YY + ZZ on 2 qubits: singlet -3, triplet +1 (x3).
    h = Heisenberg_Hamil(2, 1, 1, 1)
    vals = np.sort(h.eigenvalues())
    assert np.allclose(vals, [-3.0, 1.0, 1.0, 1.0])
    # energy_gap returns the unique *nonzero* gaps: |1 - (-3)| = 4.
    assert np.allclose(h.energy_gap(), [4.0])


def test_heisenberg_four_qubit_gaps_regression():
    # Locks the physics against the pre-refactor reference values.
    gaps = Heisenberg_Hamil(4, 1, 1, 1).energy_gap()
    assert np.allclose(np.round(gaps[:3], 4), [1.1716, 1.3643, 1.4641])


def test_ground_and_excited_state():
    h = Heisenberg_Hamil(3, 1, 1, 1)
    e0, e2, v0, v2 = h.get_ground_and_excited_state(n=2)
    full = np.sort(h.eigenvalues())
    assert np.isclose(e0, full[0])         # ground energy == smallest eigenvalue
    assert e2 >= e0
    H = h.to_dense()
    # returned vectors are genuine eigenvectors with the reported energies
    assert np.allclose(H @ v0, e0 * v0, atol=1e-6)


def test_ground_excited_index_out_of_range():
    with pytest.raises(ValueError):
        Hamiltonian(1, [("Z", [0], 1.0)]).get_ground_and_excited_state(n=2)


# --------------------------------------------------------------------------- #
#  Algebra                                                                     #
# --------------------------------------------------------------------------- #
def test_add_same_size():
    a = Hamiltonian(2, [("X", [0], 1.0)])
    b = Hamiltonian(2, [("Z", [1], 1.0)])
    s = a + b
    assert len(s) == 2 and s.nqubits == 2
    assert np.allclose(s.to_dense(), np.kron(X, I2) + np.kron(I2, Z))


def test_add_size_mismatch_raises():
    with pytest.raises(ValueError):
        Hamiltonian(2, [("X", [0], 1.0)]) + Hamiltonian(3, [("X", [0], 1.0)])


def test_tensor_product():
    a = Hamiltonian(1, [("Z", [0], 1.0)])
    b = Hamiltonian(2, [("X", [0], 1.0)])
    p = a * b
    assert p.nqubits == 3
    # b's term is shifted onto qubit 1
    assert ("X", [1], 1.0) in p.terms


# --------------------------------------------------------------------------- #
#  Subclasses                                                                  #
# --------------------------------------------------------------------------- #
def test_heisenberg_term_counts():
    open_h = Heisenberg_Hamil(4, 1, 1, 1)
    assert len(open_h) == 3 * (4 - 1)                 # 3 couplings * (n-1) bonds
    periodic_h = Heisenberg_Hamil(4, 1, 1, 1, periodic=True)
    assert len(periodic_h) == 3 * 4                   # n bonds with PBC
    full_h = Heisenberg_Hamil(4, 1, 1, 1, fully_connected=True)
    assert len(full_h) == 3 * (4 * 3 // 2)            # 3 couplings * C(4,2)


def test_ising_terms_and_signs():
    h = Ising_Hamil(3, J=2.0, transverse=0.5, longitudinal=0.3, periodic=False)
    coeffs = {(p, tuple(q)): c for p, q, c in h.terms}
    assert coeffs[("ZZ", (0, 1))] == -2.0             # -J
    assert coeffs[("X", (0,))] == -0.5                # -g
    assert coeffs[("Z", (0,))] == -0.3                # -h
    # 2 ZZ bonds (open) + 3 X + 3 Z
    assert len(h) == 2 + 3 + 3


def test_ising_transverse_field_matches_paper_model():
    # Paper Fig. 3 model: H = -J sum ZZ - d sum X (open chain).
    h = Ising_Hamil(4, J=0.1, transverse=2.0, periodic=False)
    coeffs = {(p, tuple(q)): c for p, q, c in h.terms}
    assert coeffs[("ZZ", (0, 1))] == -0.1
    assert coeffs[("X", (0,))] == -2.0
    assert len(h) == 3 + 4            # 3 ZZ bonds (open) + 4 X, no Z field
