"""Tests for algorithmic shadow spectroscopy."""

import numpy as np
import pytest

from pai_shadow.hamil import Heisenberg_Hamil
from pai_shadow.shadow_spectro import (
    Spectroscopy,
    dominant_gap,
    k_local_paulis,
    te_pai_shadow_spectroscopy,
    trotter_shadow_spectroscopy,
)


def test_k_local_paulis_count():
    assert len(k_local_paulis(3, 1)) == 3 * 3                  # 3 qubits x XYZ
    assert len(k_local_paulis(4, 2)) == 4 * 3 + 6 * 9          # 1-local + 2-local
    # every label has the right length and at most k non-identity factors
    for p in k_local_paulis(4, 2):
        assert len(p) == 4 and (4 - p.count("I")) <= 2


def test_spectroscopy_recovers_injected_frequency():
    rng = np.random.default_rng(0)
    Nt, dt, No, omega = 200, 0.1, 60, 4.36
    t = np.arange(Nt) * dt
    D = np.array([np.cos(omega * t + 2 * np.pi * rng.random()) + 0.05 * rng.standard_normal(Nt)
                  for _ in range(No)]).T
    freqs, spec = Spectroscopy(dt, cutoff=4, damping=0.05).spectrum(D, ljung=False)
    assert abs(dominant_gap(freqs, spec) - omega) < 0.4       # within frequency resolution


def test_dominant_gap():
    freqs = np.array([0.0, 1.0, 2.0, 3.0])
    spec = np.array([0.1, 0.2, 0.9, 0.3])
    assert dominant_gap(freqs, spec) == 2.0


def _heisenberg_gap_state():
    H = Heisenberg_Hamil(4, 1, 1, 1)
    e0, e1, g, x = H.get_ground_and_excited_state(n=1)
    return H, (g + x), e1 - e0                                # init = ground + excited


def test_trotter_pipeline_recovers_gap():
    np.random.seed(0)
    H, init, gap = _heisenberg_gap_state()
    times = np.arange(40) * 0.4
    freqs, spec = trotter_shadow_spectroscopy(H, init, times, n_steps=50,
                                              shadow_size=500, k=2, seed=0)
    assert abs(dominant_gap(freqs, spec) - gap) < 0.5


def test_te_pai_pipeline_recovers_gap():
    np.random.seed(0)
    H, init, gap = _heisenberg_gap_state()
    times = np.arange(40) * 0.4
    freqs, spec = te_pai_shadow_spectroscopy(H, init, times, delta=np.pi / 8,
                                             M=1000, k=2, seed=0)
    assert abs(dominant_gap(freqs, spec) - gap) < 0.6
