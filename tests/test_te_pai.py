"""Tests for the TE-PAI random-circuit generator."""

import numpy as np
import pytest

from pai_shadow.backend import get_backend
from pai_shadow.hamil import Heisenberg_Hamil, Ising_Hamil
from pai_shadow.te_pai import TEPAI
from pai_shadow.trotter import trotter_circuit


def random_state(n, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(1 << n) + 1j * rng.standard_normal(1 << n)
    return v / np.linalg.norm(v)


def test_overhead_at_least_one():
    h = Heisenberg_Hamil(3, 1, 1, 1)
    tp = TEPAI(h, delta=np.pi / 8, T=0.5, n_steps=4)
    assert tp.overhead >= 1.0


def test_sample_shapes_and_weights():
    h = Heisenberg_Hamil(2, 1, 1, 1)
    psi0 = random_state(2, seed=1)
    tp = TEPAI(h, delta=np.pi / 8, T=0.5, n_steps=4, init_state=psi0)
    circuits, weights = tp.sample(50)
    assert len(circuits) == 50 and weights.shape == (50,)
    # every weight has magnitude == overhead, sign +/- 1
    assert np.allclose(np.abs(weights), tp.overhead)
    assert set(np.sign(weights)).issubset({-1.0, 1.0})
    # init state is propagated to every circuit
    assert all(c.init_state is psi0 for c in circuits)


def test_sampled_gate_angles_are_discrete():
    h = Heisenberg_Hamil(3, 1, 1, 1)
    delta = np.pi / 8
    tp = TEPAI(h, delta=delta, T=1.0, n_steps=8)  # theta = 2*1/8 = 0.25 <= delta
    circuits, _ = tp.sample(100)
    allowed = {round(delta, 12), round(-delta, 12), round(np.pi, 12)}
    for c in circuits:
        for g in c.gates:
            assert round(g.param, 12) in allowed


def test_recommended_samples():
    h = Heisenberg_Hamil(2, 1, 1, 1)
    tp = TEPAI(h, delta=np.pi / 8, T=0.5, n_steps=4)
    assert tp.recommended_samples(0.1) == int(np.ceil((tp.overhead / 0.1) ** 2))


def test_unbiased_estimator_matches_trotter():
    """The weighted TE-PAI mean estimates the first-order Trotter expectation."""
    np.random.seed(0)
    h = Ising_Hamil(2, J=1.0, transverse=0.8, periodic=False)
    psi0 = random_state(2, seed=2)
    T, n_steps, delta = 0.6, 4, np.pi / 8
    obs = "ZZ"
    be = get_backend("qulacs")

    reference = be.expectation(trotter_circuit(h, T, n_steps, init_state=psi0), obs)

    tp = TEPAI(h, delta=delta, T=T, n_steps=n_steps, init_state=psi0)
    circuits, weights = tp.sample(15000)
    est = np.mean([w * be.expectation(c, obs) for c, w in zip(circuits, weights)])

    assert abs(est - reference) < 0.05


def test_angle_exceeds_delta_raises():
    # theta = 2*1*1.0/2 = 1.0 > delta=pi/16 -> invalid, must raise.
    h = Heisenberg_Hamil(2, 1, 1, 1)
    with pytest.raises(ValueError):
        TEPAI(h, delta=np.pi / 16, T=1.0, n_steps=2)


def test_invalid_arguments():
    h = Heisenberg_Hamil(2, 1, 1, 1)
    with pytest.raises(ValueError):
        TEPAI(h, delta=np.pi / 8, T=0.5, n_steps=0)
    tp = TEPAI(h, delta=np.pi / 8, T=0.5, n_steps=4)
    with pytest.raises(ValueError):
        tp.sample(0)
