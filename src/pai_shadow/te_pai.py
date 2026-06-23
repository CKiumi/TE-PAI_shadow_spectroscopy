"""TE-PAI: time evolution by probabilistic angle interpolation.

Each small first-order Trotter rotation ``R_P(theta)`` (with ``theta = 2|coef|dt``)
is written, via probabilistic angle interpolation, as a quasiprobability mixture
of three operations applied to qubits:

    identity,  R_P(+/-Delta),  R_P(pi)

with sampling overhead ``gamma`` per term. Sampling a circuit means, for every
(step, term), drawing one of these three options; the product over the chosen
options is an unbiased estimator of the deterministic first-order Trotter channel
(see :func:`pai_shadow.trotter.trotter_circuit`).

This module is backend-independent: :meth:`TEPAI.sample` returns lightweight
:class:`~pai_shadow.backend.circuit.Circuit` objects together with their signed
weights ``+/- gamma``. Estimating an observable ``O`` is then

    <O> ~= (1/M) * sum_s  weight_s * backend.expectation(circuit_s, O).

Fast generation: the discrete set of possible gates is precomputed once as shared
:class:`~pai_shadow.backend.circuit.Gate` objects (angles are exactly ``+/-Delta``
or ``pi``), the three-way categorical draw is vectorised over all circuits at
once, and per-circuit work is only collecting references to the precomputed gates.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .backend.circuit import Circuit, Gate
from .hamil import Hamiltonian


def _abc(theta: np.ndarray, delta: float):
    """Quasiprobability coefficients (a, b, c) of the PAI decomposition.

    ``theta`` is assumed non-negative (it is ``2|coef|dt``), so the fixed angle
    is ``+delta`` everywhere.
    """
    a = (1 + np.cos(theta) - (np.cos(delta) + 1) / np.sin(delta) * np.sin(theta)) / 2
    b = np.sin(theta) / np.sin(delta)
    c = (1 - np.cos(theta) - np.sin(theta) * np.tan(delta / 2)) / 2
    return a, b, c


class TEPAI:
    """Generator of TE-PAI random circuits for ``exp(-i H T)``.

    Parameters
    ----------
    hamil:
        Hamiltonian to evolve under.
    delta:
        Fixed PAI rotation angle ``Delta`` (smaller -> shallower, larger overhead).
    T:
        Total evolution time.
    n_steps:
        Number of (first-order) Trotter steps; step size ``dt = T / n_steps``.
    init_state:
        Optional initial state vector (little-endian) prepended to every circuit.
    """

    def __init__(self, hamil: Hamiltonian, delta: float, T: float, n_steps: int,
                 init_state: np.ndarray | None = None):
        if n_steps < 1:
            raise ValueError("n_steps must be >= 1.")
        self.nq = hamil.nqubits
        self.delta = float(delta)
        self.T = float(T)
        self.n_steps = int(n_steps)
        self.n_terms = len(hamil)
        self.init_state = init_state

        step_times = np.linspace(0, T, n_steps)
        dt = T / n_steps
        coefs = np.array([np.real(hamil.coefs(t)) for t in step_times])  # (S, K)
        angles = 2.0 * np.abs(coefs) * dt                                # theta >= 0

        # TE-PAI requires each Trotter angle to satisfy theta <= delta, otherwise
        # the angle-interpolation overhead drops below 1 and the decomposition is
        # invalid. Increase n_steps (smaller dt) or delta if this fails.
        max_angle = float(angles.max()) if angles.size else 0.0
        if max_angle > self.delta + 1e-9:
            need = int(np.ceil(max_angle / self.delta * self.n_steps))
            raise ValueError(
                f"Max Trotter angle {max_angle:.4f} exceeds delta={self.delta:.4f}; "
                f"TE-PAI needs 2|coef|*dt <= delta. Increase n_steps to >= {need}."
            )

        a, b, c = _abc(angles, self.delta)
        weights3 = np.stack([np.abs(a), np.abs(b), np.abs(c)], axis=-1)  # (S, K, 3)
        probs = weights3 / weights3.sum(axis=-1, keepdims=True)
        # cumulative thresholds: option 1 (a/identity), 2 (b/+-Delta), 3 (c/pi)
        self._cdf0 = probs[:, :, 0]
        self._cdf1 = probs[:, :, 0] + probs[:, :, 1]

        # total sampling overhead gamma = prod cos(Delta/2 - theta)/cos(Delta/2)
        gpt = np.cos(self.delta / 2 - angles) / np.cos(self.delta / 2)
        self.overhead = float(np.prod(gpt))

        # precompute shared Gate objects per (step, term)
        signs = np.sign(coefs)
        self._gate_delta: List[List[Gate]] = []
        self._gate_pi: List[List[Gate]] = []
        for i, t in enumerate(step_times):
            row_d, row_p = [], []
            for j, (pauli, qubits, _) in enumerate(hamil.get_term(t)):
                name = "R" + pauli
                qt = tuple(qubits)
                row_d.append(Gate(name, qt, signs[i, j] * self.delta))
                row_p.append(Gate(name, qt, np.pi))
            self._gate_delta.append(row_d)
            self._gate_pi.append(row_p)

    def recommended_samples(self, pai_error: float) -> int:
        """Number of circuits for a target statistical error: ``(gamma/eps)**2``."""
        return int(np.ceil((self.overhead / pai_error) ** 2))

    def sample(self, n_circuits: int) -> Tuple[List[Circuit], np.ndarray]:
        """Sample ``n_circuits`` TE-PAI circuits and their signed weights.

        Returns
        -------
        circuits:
            list of :class:`Circuit` (sparse gate lists, shared Gate objects).
        weights:
            array of ``+/- overhead`` (sign from the number of pi-flips sampled).
        """
        if n_circuits < 1:
            raise ValueError("n_circuits must be >= 1.")
        r = np.random.random((n_circuits, self.n_steps, self.n_terms))
        # val: 1 = identity, 2 = +/-Delta rotation, 3 = pi flip
        val = 1 + (r >= self._cdf0[None]).astype(np.int8) + (r >= self._cdf1[None]).astype(np.int8)

        circuits: List[Circuit] = []
        weights = np.empty(n_circuits)
        for s in range(n_circuits):
            vs = val[s]
            gates: List[Gate] = []
            sign = 1
            for i, j in np.argwhere(vs != 1):
                if vs[i, j] == 3:
                    sign = -sign
                    gates.append(self._gate_pi[i][j])
                else:
                    gates.append(self._gate_delta[i][j])
            circuits.append(Circuit(self.nq, gates, init_state=self.init_state))
            weights[s] = sign * self.overhead
        return circuits, weights
