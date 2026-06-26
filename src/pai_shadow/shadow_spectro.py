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
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
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

    def __init__(self, dt: float, cutoff: int = 4, damping: float = 0.1, pad: int = 1):
        self.dt = dt
        self.cutoff = cutoff
        self.damping = damping
        self.pad = int(pad)             # FFT zero-padding factor (>=1): finer frequency grid

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
        # Zero-pad the FFT to nfft = n_corr * pad. Padding does not add real spectral
        # resolution (that is fixed by total_time) but sinc-interpolates the spectrum
        # onto a finer frequency grid, so the plotted curve is smooth instead of
        # coarsely sampled. pad=1 reproduces the original (n_corr-point) grid exactly.
        nfft = n_corr * self.pad
        data = np.array([
            [np.fft.fft(self._xcorr(vectors[a], vectors[b]), n=nfft) for b in range(K)]
            for a in range(K)
        ])  # (K, K, nfft)
        spectrum = np.array([
            np.max(np.linalg.svd(data[:, :, f], compute_uv=False)) for f in range(nfft)
        ])
        # max frequency is the Nyquist 2*pi*n_corr/total_time = 2*pi/dt; pad=1 -> identical
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
def _spectroscopy(dt, cutoff, damping, pad=1):
    return Spectroscopy(dt, cutoff, damping, pad)


def _progress(done: int, total: int, t0: float, label: str | None) -> None:
    """One-line progress to stderr (overwritten in a TTY, appended in a log)."""
    elapsed = time.perf_counter() - t0
    eta = elapsed / done * (total - done) if done else 0.0
    tag = f"{label} " if label else ""
    end = "\n" if (done == total or not sys.stderr.isatty()) else "\r"
    sys.stderr.write(
        f"  [{tag}{done:>4}/{total} time points  {elapsed:6.1f}s elapsed  ~{eta:6.1f}s left]{end}"
    )
    sys.stderr.flush()


def _data_matrix(worker: Callable, tasks: List[tuple], n_jobs,
                 costs: Sequence[float] | None = None,
                 label: str | None = None) -> np.ndarray:
    """Run the per-time-point ``worker`` over ``tasks`` and stack the rows.

    Time points are independent, so they are distributed across ``n_jobs`` worker
    processes (``n_jobs=None`` uses all CPU cores; ``n_jobs=1`` runs serially).

    Each task carries its own seed, so the result is identical regardless of the
    execution order. When ``costs`` is given, tasks are dispatched heaviest-first
    (longest-processing-time scheduling) so a slow task does not strand a worker
    at the tail; the rows are reordered back to the original time order, so the
    output is bit-for-bit the same as serial execution.

    Progress (time points completed, elapsed, ETA) is reported to stderr as each
    point finishes; ``label`` tags the line so concurrent curves are
    distinguishable in a log.
    """
    n_jobs = n_jobs or os.cpu_count() or 1
    n_jobs = max(1, min(n_jobs, len(tasks)))
    total = len(tasks)
    t0 = time.perf_counter()
    if n_jobs == 1:
        rows = []
        for k, task in enumerate(tasks, 1):
            rows.append(worker(task))
            _progress(k, total, t0, label)
        return np.array(rows)

    order = list(range(len(tasks)))
    if costs is not None:
        order.sort(key=lambda i: costs[i], reverse=True)
    rows: list = [None] * len(tasks)
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        futs = {pool.submit(worker, tasks[i]): i for i in order}
        for done, fut in enumerate(as_completed(futs), 1):
            rows[futs[fut]] = fut.result()
            _progress(done, total, t0, label)
    return np.array(rows)


def _trotter_row(packed):
    hamil, init_state, t, n_steps, observables, shadow_size, noise, seed, density = packed
    shadow = ClassicalShadow(seed=seed)
    circ = trotter_circuit(hamil, t, n_steps, init_state=init_state)
    factors = shadow.snapshots(circ, shadow_size, noise=noise, density=density)
    return shadow.expectations(observables, factors)


def _te_pai_row(packed):
    hamil, init_state, t, n_steps, delta, M, n_shots, noise, observables, seed, exact = packed
    ss = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
    s_sample, s_shadow = ss.spawn(2)
    if n_steps is None:
        # adaptive: minimal steps so the angle sits at delta (overhead ~1).
        cmax = max((abs(np.real(c)) for _, _, c in hamil.get_term(0.0)), default=1.0)
        n_steps = max(1, int(np.ceil(2 * cmax * t / delta)))
    tp = TEPAI(hamil, delta, t, n_steps, init_state=init_state)
    circuits, weights = tp.sample(M, rng=np.random.default_rng(s_sample))
    if exact:
        # exact (infinite-shot) per-circuit expectations: removes the classical-
        # shadow 3**k single-shot variance, keeping only the gamma**2/M TE-PAI
        # sampling variance -- the clean regime of the paper figures.
        from .circuit import make_observables, weighted_exact_data_row
        obs = make_observables(observables, hamil.nqubits)
        return weighted_exact_data_row(circuits, weights, obs, noise=noise)
    shadow = ClassicalShadow(seed=s_shadow)
    factors = shadow.snapshots_per_circuit(circuits, n_shots, noise=noise)
    # flatten (M, n_shots) snapshots; repeat each circuit weight n_shots times
    factors = factors.reshape(M * n_shots, hamil.nqubits, 3)
    w = np.repeat(weights, n_shots)
    return shadow.expectations(observables, factors, weights=w)


def trotter_shadow_spectroscopy(
    hamil, init_state, times, n_steps=None, shadow_size=1000, k=3,
    trotter_step=None, noise=None, density=False,
    seed=None, ljung=True, cutoff=4, damping=0.1, pad=1, n_jobs=1, label=None,
    return_data=False,
):
    """Shadow spectroscopy with deterministic Trotter time evolution.

    The Trotter step count for evolving to ``t`` is set by ``trotter_step`` (a
    fixed step size ``dt_T``: ``n = round(t / dt_T)`` steps, i.e. constant width)
    or by a fixed ``n_steps`` (same count for every time point, width ``t/n``).

    With a :class:`NoiseSpec` ``noise``, each shadow snapshot is an independent
    noisy (trajectory) single shot of the Trotter circuit, modelling gate noise
    (statistically identical to sampling from the exact noisy density matrix).

    ``n_jobs`` worker processes split the (independent) time points; ``None``
    uses all CPU cores, ``1`` (default) runs serially.
    """
    def steps_for(t):
        if trotter_step is not None:
            return max(1, int(round(t / trotter_step)))
        return n_steps

    observables = k_local_paulis(hamil.nqubits, k)
    point_seeds = np.random.SeedSequence(seed).spawn(len(times))
    tasks = [(hamil, init_state, float(t), steps_for(float(t)), observables, shadow_size, noise, ps, density)
             for t, ps in zip(times, point_seeds)]
    D = _data_matrix(_trotter_row, tasks, n_jobs, costs=list(times), label=label)
    dt = float(times[1] - times[0])
    freqs, intensity = _spectroscopy(dt, cutoff, damping, pad).spectrum(D, ljung)
    # D is the (Nt, No) data matrix *before* any post-processing -- return it so the
    # spectrum can be recomputed at other damping/cutoff without re-simulating.
    return (freqs, intensity, D) if return_data else (freqs, intensity)


def te_pai_shadow_spectroscopy(
    hamil, init_state, times, delta, M, n_steps=None,
    trotter_step=None, n_trotter_max=None, k=3, n_shots=1, noise=None, exact=False,
    seed=None, ljung=True, cutoff=4, damping=0.1, pad=1, n_jobs=1, label=None,
    return_data=False,
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
    tasks = [(hamil, init_state, float(t), steps_for(float(t)), delta, M, n_shots, noise, observables, ps, exact)
             for t, ps in zip(times, point_seeds)]
    D = _data_matrix(_te_pai_row, tasks, n_jobs, costs=list(times), label=label)
    freqs, intensity = _spectroscopy(dt, cutoff, damping, pad).spectrum(D, ljung)
    # D is the (Nt, No) data matrix *before* post-processing -- return it so the
    # spectrum can be recomputed at other damping/cutoff without re-simulating.
    return (freqs, intensity, D) if return_data else (freqs, intensity)
