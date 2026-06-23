"""Tests for the qulacs backend.

The qulacs statevector/expectation results are checked against an independent
dense numpy simulator (``dense_statevector``) that interprets the Circuit IR
directly, so the backend is validated without relying on a second simulator.
"""

import numpy as np
import pytest
from scipy.linalg import expm

from pai_shadow.backend import Circuit, NoiseSpec, get_backend

H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

# --------------------------------------------------------------------------- #
#  Independent dense reference simulator (little-endian, qubit 0 = LSB)         #
# --------------------------------------------------------------------------- #
_P = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}
_STATIC = {
    "H": H,
    "S": np.array([[1, 0], [0, 1j]], dtype=complex),
    "X": _P["X"], "Y": _P["Y"], "Z": _P["Z"],
}
_ROT_AXIS = {"RX": "X", "RY": "Y", "RZ": "Z",
             "RXX": "X", "RYY": "Y", "RZZ": "Z"}


def _embed(single_ops: dict, n: int) -> np.ndarray:
    """Dense operator on ``n`` qubits from a {qubit: 2x2} map (little-endian)."""
    mats = [_P["I"]] * n
    for q, m in single_ops.items():
        mats[q] = m
    op = mats[n - 1]
    for q in range(n - 2, -1, -1):
        op = np.kron(op, mats[q])
    return op


def dense_statevector(circ: Circuit) -> np.ndarray:
    n = circ.num_qubits
    if circ.init_state is not None:
        psi = np.asarray(circ.init_state, dtype=complex).copy()
    else:
        psi = np.zeros(1 << n, dtype=complex)
        psi[0] = 1.0
    for g in circ.gates:
        if g.name in _STATIC:
            op = _embed({g.qubits[0]: _STATIC[g.name]}, n)
        elif g.name == "U":
            op = _embed({g.qubits[0]: g.matrix}, n)
        elif g.name in _ROT_AXIS:
            P = _P[_ROT_AXIS[g.name]]
            gen = _embed({q: P for q in g.qubits}, n)
            op = expm(-1j * g.param / 2 * gen)
        else:
            raise AssertionError(f"unknown gate {g.name}")
        psi = op @ psi
    return psi


def dense_expectation(circ: Circuit, pauli: str) -> float:
    n = circ.num_qubits
    psi = dense_statevector(circ)
    P = _embed({q: _P[p] for q, p in enumerate(pauli)}, n)
    return float(np.real(np.vdot(psi, P @ psi)))


# --------------------------------------------------------------------------- #
def sample_circuit() -> Circuit:
    """A circuit exercising every gate type plus a custom init state."""
    init = np.array([1, 0, 0, 1, 0, 0, 0, 0], dtype=complex)
    init = init / np.linalg.norm(init)
    c = Circuit(3, init_state=init)
    c.h(0).rx(1, 0.7).rz(2, -0.4)
    c.rxx(0, 1, 1.1).ryy(1, 2, 0.5).rzz(0, 2, -0.9).s(2)
    c.unitary(1, H)  # shadow-Clifford-like custom 1q unitary
    return c


def fidelity(a, b):
    return abs(np.vdot(a, b)) ** 2  # equal states up to global phase -> 1


def test_statevector_matches_dense():
    c = sample_circuit()
    sv = get_backend("qulacs").statevector(c)
    assert np.isclose(fidelity(sv, dense_statevector(c)), 1.0, atol=1e-10)


@pytest.mark.parametrize("pauli", ["ZZZ", "XIY", "IZX", "YYI", "XXX", "III"])
def test_expectation_matches_dense(pauli):
    c = sample_circuit()
    e_ql = get_backend("qulacs").expectation(c, pauli)
    assert abs(e_ql - dense_expectation(c, pauli)) < 1e-9


def test_deterministic_sampling_and_endianness():
    # X on qubit 0 of a 2-qubit register -> outcome qubit1=0, qubit0=1 -> "01".
    c = Circuit(2)
    c.x(0)
    out = get_backend("qulacs").sample(c, shots=16)
    assert out == ["01"] * 16


def test_expectation_known_value():
    # <Z> on |1> = -1 ; <X> on H|0> = +1.
    be = get_backend("qulacs")
    c1 = Circuit(1)
    c1.x(0)
    assert np.isclose(be.expectation(c1, "Z"), -1.0, atol=1e-9)
    c2 = Circuit(1)
    c2.h(0)
    assert np.isclose(be.expectation(c2, "X"), 1.0, atol=1e-9)


def test_noisy_sampling_runs():
    ns = NoiseSpec(p1=1e-2, p2=5e-2)
    be = get_backend("qulacs", noise=ns)
    out = be.sample(sample_circuit(), shots=8)
    assert len(out) == 8
    assert all(len(b) == 3 and set(b) <= {"0", "1"} for b in out)


def test_noise_kinds_physics():
    c = Circuit(1).rx(0, 0.0)  # stays |0>; noise attaches to the RX gate
    # bit flip on |0>: <Z> = 1 - 2p
    be = get_backend("qulacs", noise=NoiseSpec(p1=0.25, kind="bitflip", one_qubit_gates=("RX",)))
    z = np.mean([1 - 2 * int(b[-1]) for b in be.sample(c, 4000)])
    assert abs(z - 0.5) < 0.1
    # phase flip on |0>: leaves Z populations unchanged -> <Z> ~ 1
    be = get_backend("qulacs", noise=NoiseSpec(p1=0.4, kind="phaseflip", one_qubit_gates=("RX",)))
    z = np.mean([1 - 2 * int(b[-1]) for b in be.sample(c, 2000)])
    assert z > 0.95
    # depolarizing pulls <Z> toward 0
    be = get_backend("qulacs", noise=NoiseSpec(p1=0.5, kind="depolarizing", one_qubit_gates=("RX",)))
    z = np.mean([1 - 2 * int(b[-1]) for b in be.sample(c, 4000)])
    assert z < 0.95


def test_invalid_noise_kind():
    with pytest.raises(ValueError):
        NoiseSpec(p1=0.1, kind="banana")


def test_noisy_expectation_matches_sampling():
    # exact noisy expectation (density matrix) ~= noisy sampling mean
    c = Circuit(2).h(0).rzz(0, 1, 0.7).rx(1, 0.5)
    be = get_backend("qulacs", noise=NoiseSpec(p1=2e-3, p2=2e-2, kind="depolarizing"))
    exact_noisy = be.expectation(c, "ZZ")
    bits = be.sample(c, 8000)
    sampled = np.mean([(1 - 2 * int(b[-1])) * (1 - 2 * int(b[-2])) for b in bits])
    assert abs(exact_noisy - sampled) < 0.05


def test_circuit_depth():
    c = Circuit(3)
    c.h(0).h(1).h(2)          # layer 1 (parallel)
    c.rzz(0, 1, 0.1)          # layer 2
    c.rzz(1, 2, 0.1)          # layer 3 (shares qubit 1)
    assert c.depth() == 3
    assert len(c) == 5
