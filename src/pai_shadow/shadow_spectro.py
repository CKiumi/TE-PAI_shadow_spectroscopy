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
from typing import List, Sequence, Tuple

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


def trotter_shadow_spectroscopy(
    hamil, init_state, times, n_steps, shadow_size, k=3,
    seed=None, ljung=True, cutoff=4, damping=0.1,
):
    """Shadow spectroscopy with deterministic Trotter time evolution."""
    shadow = ClassicalShadow(seed=seed)
    observables = k_local_paulis(hamil.nqubits, k)
    D = np.empty((len(times), len(observables)))
    for i, t in enumerate(times):
        circ = trotter_circuit(hamil, t, n_steps, init_state=init_state)
        factors = shadow.snapshots(circ, shadow_size)
        D[i] = shadow.expectations(observables, factors)
    dt = float(times[1] - times[0])
    return _spectroscopy(dt, cutoff, damping).spectrum(D, ljung)


def te_pai_shadow_spectroscopy(
    hamil, init_state, times, delta, M, k=3,
    seed=None, ljung=True, cutoff=4, damping=0.1,
):
    """Shadow spectroscopy with shallow TE-PAI random circuits.

    Each sampled TE-PAI circuit gets a single shadow snapshot; the per-observable
    estimate is the quasiprobability-weighted snapshot mean. The number of Trotter
    steps is chosen per time so the angle satisfies ``2|coef|*dt ~ delta`` (which
    minimises the TE-PAI sampling overhead).
    """
    shadow = ClassicalShadow(seed=seed)
    observables = k_local_paulis(hamil.nqubits, k)
    cmax = max((abs(np.real(c)) for _, _, c in hamil.get_term(0.0)), default=1.0)
    D = np.empty((len(times), len(observables)))
    for i, t in enumerate(times):
        n_steps = max(1, int(np.ceil(2 * cmax * t / delta)))
        tp = TEPAI(hamil, delta, t, n_steps, init_state=init_state)
        circuits, weights = tp.sample(M)
        factors = shadow.snapshots_of_circuits(circuits)
        D[i] = shadow.expectations(observables, factors, weights=weights)
    dt = float(times[1] - times[0])
    return _spectroscopy(dt, cutoff, damping).spectrum(D, ljung)
