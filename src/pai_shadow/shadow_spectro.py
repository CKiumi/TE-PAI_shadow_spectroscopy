"""Algorithmic shadow spectroscopy (Chan, Meister, Goh & Koczor 2025).

Estimate the energy gaps ``Delta E = |E_a - E_b|`` of a Hamiltonian from the
time signal of randomized (classical-shadow) measurements:

1. Evolve the initial state to a grid of times ``t_k = k*dt``.
2. At each time, take classical-shadow snapshots and estimate many k-local Pauli
   observables, building a data matrix ``D`` of shape ``(Nt, No)``.
3. Spectral analysis of ``D`` (standardise -> dominant temporal modes ->
   cross-correlation + SVD over frequency) yields a spectrum whose peaks sit at
   the energy gaps.

Two front ends share the spectral analysis:
``trotter_shadow_spectroscopy`` (deterministic Trotter evolution) and
``te_pai_shadow_spectroscopy`` (shallow TE-PAI random circuits).
"""

from __future__ import annotations

import itertools
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, List, Sequence, Tuple

import numpy as np

from .classical_shadow import ClassicalShadow
from .trotter import trotter_circuit
from .te_pai import TEPAI


def k_local_paulis(num_qubits: int, k: int) -> List[str]:
    """All Pauli strings with 1..k non-identity X/Y/Z factors."""
    out: List[str] = []
    for kk in range(1, k + 1):
        for positions in itertools.combinations(range(num_qubits), kk):
            for combo in itertools.product("XYZ", repeat=kk):
                s = ["I"] * num_qubits
                for pos, p in zip(positions, combo):
                    s[pos] = p
                out.append("".join(s))
    return out


class Spectroscopy:
    """Spectral analysis of a time x observable data matrix.

    Parameters
    ----------
    dt:
        Time spacing between rows of the data matrix.
    cutoff:
        Number of dominant temporal modes (eigenvectors) to keep.
    damping:
        Exponential damping rate applied along time to suppress edge effects.
    """

    def __init__(self, dt: float, cutoff: int = 4, damping: float = 0.1):
        self.dt = dt
        self.cutoff = cutoff
        self.damping = damping

    @staticmethod
    def _standardize(D: np.ndarray) -> np.ndarray:
        mu = D.mean(axis=0)
        sd = D.std(axis=0)
        sd = np.where(sd < 1e-9, 1.0, sd)   # floor: constant columns -> 0, no blow-up
        return (D - mu) / sd

    @staticmethod
    def _ljung_box(X: np.ndarray, ratio: float = 5.0) -> np.ndarray:
        """Keep the most autocorrelated columns (most signal-like)."""
        from statsmodels.stats.diagnostic import acorr_ljungbox

        No = X.shape[1]
        pvals = np.ones(No)
        for j in range(No):
            try:
                pvals[j] = acorr_ljungbox(X[:, j], lags=[1], return_df=True)["lb_pvalue"].iloc[0]
            except Exception:
                pvals[j] = 1.0
        keep = min(No, max(100, int(No * ratio / 100)))
        return X[:, np.argsort(pvals)[:keep]]

    @staticmethod
    def _xcorr(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        n = len(x)
        return np.array([np.mean(x[: n - m] * y[m:]) for m in range(1, n)])

    def _spectral_cross_correlation(self, vectors: Sequence[np.ndarray]):
        K = len(vectors)
        n_corr = len(vectors[0]) - 1
        total_time = n_corr * self.dt
        data = np.array([
            [np.fft.fft(self._xcorr(vectors[a], vectors[b])) for b in range(K)]
            for a in range(K)
        ])  # (K, K, n_corr)
        spectrum = np.array([
            np.max(np.linalg.svd(data[:, :, f], compute_uv=False)) for f in range(n_corr)
        ])
        freqs = np.linspace(0, 2 * np.pi * n_corr / total_time, len(spectrum))
        half = len(spectrum) // 2
        return freqs[:half], spectrum[:half]

    def spectrum(self, D: np.ndarray, ljung: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(frequencies, intensity)`` from a ``(Nt, No)`` data matrix."""
        D = np.asarray(D, dtype=float)
        Nt = D.shape[0]
        X = self._standardize(D)
        if ljung:
            X = self._ljung_box(X)
        X = X * np.exp(-self.damping * np.arange(Nt) * self.dt)[:, None]
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            C = X @ X.T                               # (Nt, Nt) temporal correlation
        vals, vecs = np.linalg.eigh(C)
        order = np.argsort(vals)[::-1][: self.cutoff]
        vectors = [vecs[:, i] for i in order]
        return self._spectral_cross_correlation(vectors)


def dominant_gap(frequencies: np.ndarray, intensity: np.ndarray) -> float:
    """Frequency of the largest spectral peak (the dominant energy gap)."""
    return float(frequencies[np.argmax(np.abs(intensity))])


# --------------------------------------------------------------------------- #
#  Front ends: build the data matrix, then run the spectral analysis           #
# --------------------------------------------------------------------------- #
def _spectroscopy(dt, cutoff, damping):
    return Spectroscopy(dt, cutoff, damping)


def _data_matrix(worker: Callable, tasks: List[tuple], n_jobs,
                 costs: Sequence[float] | None = None) -> np.ndarray:
    """Run the per-time-point ``worker`` over ``tasks`` and stack the rows.

    Time points are independent, so they are distributed across ``n_jobs`` worker
    processes (``n_jobs=None`` uses all CPU cores; ``n_jobs=1`` runs serially).

    Each task carries its own seed, so the result is identical regardless of the
    execution order. When ``costs`` is given, tasks are dispatched heaviest-first
    (longest-processing-time scheduling) so a slow task does not strand a worker
    at the tail; the rows are reordered back to the original time order, so the
    output is bit-for-bit the same as serial execution.
    """
    n_jobs = n_jobs or os.cpu_count() or 1
    n_jobs = max(1, min(n_jobs, len(tasks)))
    if n_jobs == 1:
        return np.array([worker(task) for task in tasks])

    order = list(range(len(tasks)))
    if costs is not None:
        order.sort(key=lambda i: costs[i], reverse=True)
    rows: list = [None] * len(tasks)
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        for i, row in zip(order, pool.map(worker, [tasks[i] for i in order])):
            rows[i] = row
    return np.array(rows)


def _trotter_row(packed):
    hamil, init_state, t, n_steps, observables, shadow_size, noise, seed = packed
    shadow = ClassicalShadow(seed=seed)
    circ = trotter_circuit(hamil, t, n_steps, init_state=init_state)
    factors = shadow.snapshots(circ, shadow_size, noise=noise)
    return shadow.expectations(observables, factors)


def _te_pai_row(packed):
    hamil, init_state, t, n_steps, delta, M, n_shots, noise, observables, seed = packed
    ss = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
    s_sample, s_shadow = ss.spawn(2)
    shadow = ClassicalShadow(seed=s_shadow)
    if n_steps is None:
        # adaptive: minimal steps so the angle sits at delta (overhead ~1).
        cmax = max((abs(np.real(c)) for _, _, c in hamil.get_term(0.0)), default=1.0)
        n_steps = max(1, int(np.ceil(2 * cmax * t / delta)))
    tp = TEPAI(hamil, delta, t, n_steps, init_state=init_state)
    circuits, weights = tp.sample(M, rng=np.random.default_rng(s_sample))
    factors = shadow.snapshots_per_circuit(circuits, n_shots, noise=noise)
    # flatten (M, n_shots) snapshots; repeat each circuit weight n_shots times
    factors = factors.reshape(M * n_shots, hamil.nqubits, 3)
    w = np.repeat(weights, n_shots)
    return shadow.expectations(observables, factors, weights=w)


def trotter_shadow_spectroscopy(
    hamil, init_state, times, n_steps, shadow_size, k=3, noise=None,
    seed=None, ljung=True, cutoff=4, damping=0.1, n_jobs=1,
):
    """Shadow spectroscopy with deterministic Trotter time evolution.

    With a :class:`NoiseSpec` ``noise``, each shadow snapshot is an independent
    noisy (trajectory) single shot of the Trotter circuit, modelling gate noise.

    ``n_jobs`` worker processes split the (independent) time points; ``None``
    uses all CPU cores, ``1`` (default) runs serially.
    """
    observables = k_local_paulis(hamil.nqubits, k)
    point_seeds = np.random.SeedSequence(seed).spawn(len(times))
    tasks = [(hamil, init_state, float(t), n_steps, observables, shadow_size, noise, ps)
             for t, ps in zip(times, point_seeds)]
    D = _data_matrix(_trotter_row, tasks, n_jobs, costs=list(times))
    dt = float(times[1] - times[0])
    return _spectroscopy(dt, cutoff, damping).spectrum(D, ljung)


def te_pai_shadow_spectroscopy(
    hamil, init_state, times, delta, M, n_steps=None,
    trotter_step=None, n_trotter_max=None, k=3, n_shots=1, noise=None,
    seed=None, ljung=True, cutoff=4, damping=0.1, n_jobs=1,
):
    """Shadow spectroscopy with shallow TE-PAI random circuits.

    ``M`` TE-PAI circuits are sampled per time point and each is measured
    ``n_shots`` times (the total circuit-execution budget is ``M * n_shots``); the
    per-observable estimate is the quasiprobability-weighted snapshot mean.

    The number of first-order Trotter steps for evolving to ``t`` is set by:

    * ``trotter_step`` (a step size ``dt_T``) -- ``n = round(t / dt_T)`` steps,
      capped at ``n_trotter_max`` (the reference implementation's scheme).
    * ``n_steps`` (an int) -- that many steps for every time point.
    * neither -- adaptive: minimal steps so the angle equals ``delta`` (overhead
      ``gamma`` ~ 1, i.e. TE-PAI reduces to deterministic Trotter).

    The angle ``2|coef|*t/n`` must stay ``<= delta`` or :class:`TEPAI` raises. The
    sampling overhead grows like ``exp(2 t ||H||_1 tan(delta/2))``, so a small
    ``delta`` and a moderate total time keep it manageable.

    With a :class:`NoiseSpec` ``noise``, each shadow snapshot is an independent
    noisy (trajectory) single shot; because TE-PAI circuits are much shallower
    than Trotter's, they accumulate far less gate noise (paper Fig. 2).

    ``n_jobs`` worker processes split the (independent) time points; ``None``
    uses all CPU cores, ``1`` (default) runs serially.
    """
    def steps_for(t):
        if trotter_step is not None:
            n = max(1, int(round(t / trotter_step)))
            return min(n, n_trotter_max) if n_trotter_max else n
        return n_steps

    observables = k_local_paulis(hamil.nqubits, k)
    point_seeds = np.random.SeedSequence(seed).spawn(len(times))
    dt = float(times[1] - times[0])
    tasks = [(hamil, init_state, float(t), steps_for(float(t)), delta, M, n_shots, noise, observables, ps)
             for t, ps in zip(times, point_seeds)]
    D = _data_matrix(_te_pai_row, tasks, n_jobs, costs=list(times))
    return _spectroscopy(dt, cutoff, damping).spectrum(D, ljung)
