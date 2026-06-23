"""Cross-backend agreement tests for the qiskit and qulacs bindings."""

import numpy as np
import pytest

from pai_shadow.backend import Circuit, NoiseSpec, get_backend

H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


def sample_circuit() -> Circuit:
    """A circuit exercising every gate type plus a custom init state."""
    init = np.array([1, 0, 0, 1, 0, 0, 0, 0], dtype=complex)
    c = Circuit(3, init_state=init)
    c.h(0).rx(1, 0.7).rz(2, -0.4)
    c.rxx(0, 1, 1.1).ryy(1, 2, 0.5).rzz(0, 2, -0.9).s(2)
    c.unitary(1, H)  # shadow-Clifford-like custom 1q unitary
    return c


def fidelity(a, b):
    return abs(np.vdot(a, b)) ** 2  # equal states up to global phase -> 1


def test_statevector_agreement():
    c = sample_circuit()
    sv_qk = get_backend("qiskit").statevector(c)
    sv_ql = get_backend("qulacs").statevector(c)
    assert np.isclose(fidelity(sv_qk, sv_ql), 1.0, atol=1e-10)


@pytest.mark.parametrize("pauli", ["ZZZ", "XIY", "IZX", "YYI", "XXX", "III"])
def test_expectation_agreement(pauli):
    c = sample_circuit()
    e_qk = get_backend("qiskit").expectation(c, pauli)
    e_ql = get_backend("qulacs").expectation(c, pauli)
    assert abs(e_qk - e_ql) < 1e-9


@pytest.mark.parametrize("name", ["qiskit", "qulacs"])
def test_deterministic_sampling_and_endianness(name):
    # X on qubit 0 of a 2-qubit register -> outcome qubit1=0, qubit0=1 -> "01".
    c = Circuit(2)
    c.x(0)
    out = get_backend(name).sample(c, shots=16)
    assert out == ["01"] * 16


@pytest.mark.parametrize("name", ["qiskit", "qulacs"])
def test_expectation_known_value(name):
    # <Z> on |1> = -1 ; <X> on H|0> = +1.
    be = get_backend(name)
    c1 = Circuit(1)
    c1.x(0)
    assert np.isclose(be.expectation(c1, "Z"), -1.0, atol=1e-9)
    c2 = Circuit(1)
    c2.h(0)
    assert np.isclose(be.expectation(c2, "X"), 1.0, atol=1e-9)


@pytest.mark.parametrize("name", ["qiskit", "qulacs"])
def test_noisy_sampling_runs(name):
    ns = NoiseSpec(p1=1e-2, p2=5e-2)
    be = get_backend(name, noise=ns)
    out = be.sample(sample_circuit(), shots=8)
    assert len(out) == 8
    assert all(len(b) == 3 and set(b) <= {"0", "1"} for b in out)


def test_circuit_depth():
    c = Circuit(3)
    c.h(0).h(1).h(2)          # layer 1 (parallel)
    c.rzz(0, 1, 0.1)          # layer 2
    c.rzz(1, 2, 0.1)          # layer 3 (shares qubit 1)
    assert c.depth() == 3
    assert len(c) == 5
