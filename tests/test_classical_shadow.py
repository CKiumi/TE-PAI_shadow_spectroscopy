"""Tests for random single-qubit Clifford classical shadows."""

import numpy as np
import pytest

from pai_shadow.circuit import Circuit
from pai_shadow.classical_shadow import ClassicalShadow


def test_snapshot_factor_values():
    cs = ClassicalShadow(seed=5)
    f = cs.snapshots(Circuit(3).h(0), 50)
    # each (snapshot, qubit) measures exactly one axis: one factor in {-3,+3}, rest 0
    assert set(np.unique(f)).issubset({-3.0, 0.0, 3.0})
    assert np.all((np.abs(f) > 1e-9).sum(axis=2) == 1)


def test_known_single_qubit_states():
    cs = ClassicalShadow(seed=0)
    # |+> = H|0>:  <X> = 1, <Z> = 0
    f = cs.snapshots(Circuit(1).h(0), 6000)
    assert abs(cs.expectation("X", f) - 1.0) < 0.1
    assert abs(cs.expectation("Z", f) - 0.0) < 0.1
    # |1> = X|0>:  <Z> = -1
    f1 = cs.snapshots(Circuit(1).x(0), 6000)
    assert abs(cs.expectation("Z", f1) + 1.0) < 0.1


def test_identity_observable_is_one():
    cs = ClassicalShadow(seed=1)
    f = cs.snapshots(Circuit(2).h(0), 200)
    assert cs.expectation("II", f) == 1.0


def test_unbiased_versus_exact():
    cs = ClassicalShadow(seed=2)
    c = Circuit(3)
    c.h(0).rxx(0, 1, 0.7).ryy(1, 2, 0.5).rz(2, 0.3)
    f = cs.snapshots(c, 12000)
    for P in ["ZII", "XYI", "IZX", "ZZZ"]:
        assert abs(cs.expectation(P, f) - c.expectation(P)) < 0.12, P


def test_expectations_batch_matches_single():
    cs = ClassicalShadow(seed=3)
    f = cs.snapshots(Circuit(2).h(0).rzz(0, 1, 0.4), 1500)
    paulis = ["ZI", "IZ", "XX"]
    arr = cs.expectations(paulis, f)
    assert arr.shape == (3,)
    assert np.allclose(arr, [cs.expectation(p, f) for p in paulis])


def test_snapshots_shape():
    cs = ClassicalShadow(seed=4)
    f = cs.snapshots(Circuit(5).h(0), 100)
    assert f.shape == (100, 5, 3)
