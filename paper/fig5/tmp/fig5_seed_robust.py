"""Seed-robustness check for the fig5 TE-PAI-vs-Trotter comparison (quick condition).

Question: in the nice quick figure (N=5, n_t=70, M=4000, depolarizing p2=4e-4) the
noisy TE-PAI spectrum keeps its dominant peak AT a gap while noisy Trotter collapses to
a low-frequency mode. Is that seed-independent, or a favourable realization?

Here we hold the quick condition fixed and sweep the TE-PAI RNG seed over 0..4, running
the NOISY curves of both methods, and report the dominant-peak frequency of each. If
TE-PAI's dominant peak stays near a gap across seeds -> robust; if it flips to low
frequency for some seeds -> the clean figure was a lucky draw.

Run:  uv run python paper/fig5/fig5_seed_robust.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fig5_trotter_vs_tepai as f5
from pai_shadow.hamil import Heisenberg_Hamil
from pai_shadow.shadow_spectro import dominant_gap, te_pai_shadow_spectroscopy, trotter_shadow_spectroscopy

SEEDS = [0, 1, 2, 3, 4]
N_T, T_MAX, M = 70, 6.0, 4000        # the "quick" condition
NJ = os.cpu_count()
HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    H = Heisenberg_Hamil(f5.N_QUBITS, 1.0, 1.0, 1.0)
    E, V = H.eigh()
    levels, idx = np.unique(np.round(E, 6), return_index=True)
    sel, sel_E = idx[list(f5.LEVELS)], levels[list(f5.LEVELS)]
    nlev = len(f5.LEVELS); w = np.ones(nlev); w[1:] = f5.EPS
    psi = (V[:, sel] * w).sum(axis=1); psi /= np.linalg.norm(psi)
    dom = [round(float(sel_E[k] - sel_E[0]), 4) for k in range(1, nlev)]
    cross = sorted({round(float(sel_E[j] - sel_E[i]), 4) for i in range(1, nlev) for j in range(i + 1, nlev)})

    times = np.linspace(0, T_MAX, N_T)
    DT = T_MAX / f5.N_DIV
    dt = float(times[1] - times[0])
    print(f"[robust] N={f5.N_QUBITS} gaps={dom} n_t={N_T} M={M} p2={f5.NOISE.p2:.0e}  seeds={SEEDS}", flush=True)

    results = []
    for s in SEEDS:
        t0 = time.perf_counter()
        frT, spT = te_pai_shadow_spectroscopy(
            H, psi, times, delta=f5.DELTA, M=M, n_shots=1, trotter_step=DT, n_trotter_max=None,
            k=f5.K_LOCAL, noise=f5.NOISE, damping=f5.DAMPING, pad=f5.PAD, seed=s, n_jobs=NJ, label=f"TE-PAI s{s}")
        frR, spR = trotter_shadow_spectroscopy(
            H, psi, times, trotter_step=DT, shadow_size=M, k=f5.K_LOCAL, noise=f5.NOISE, density=True,
            damping=f5.DAMPING, pad=f5.PAD, seed=s + 100, n_jobs=NJ, label=f"Trotter s{s}")
        pkT, pkR = dominant_gap(frT, spT), dominant_gap(frR, spR)
        near = lambda p: min(abs(p - g) for g in dom) < 0.8
        print(f"[robust] seed {s}: TE-PAI noisy peak@{pkT:.2f} ({'GAP' if near(pkT) else 'low-freq'})  "
              f"Trotter noisy peak@{pkR:.2f} ({'GAP' if near(pkR) else 'low-freq'})  "
              f"({time.perf_counter()-t0:.0f}s)", flush=True)
        results.append((s, frT, spT, frR, spR, pkT, pkR))

    _plot(results, dom, cross, dt)


def _plot(results, dom, cross, dt) -> None:
    n = len(results); ncol = min(5, n); nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 4.2 * nrow), squeeze=False)
    axes = axes.ravel()
    near = lambda p: min(abs(p - g) for g in dom) < 0.8
    for ax, (s, frT, spT, frR, spR, pkT, pkR) in zip(axes, results):
        yT, yR = np.abs(spT), np.abs(spR)
        ax.plot(frT, yT / (yT.max() or 1), color="red", lw=1.7, label="TE-PAI, noisy")
        ax.plot(frR, yR / (yR.max() or 1), color="blue", lw=1.5, label="Trotter, noisy")
        for j, g in enumerate(dom):
            ax.axvline(g, color="black", ls="--", lw=1.0, label="gaps" if j == 0 else None)
        for g in cross:
            ax.axvline(g, color="0.75", ls=":", lw=0.8)
        ax.set_xlim(0, max(dom + cross) + 1); ax.set_ylim(0, 1.05)
        ax.grid(True, ls="--", lw=0.5, alpha=0.5)
        ok = "TE-PAI=GAP" if near(pkT) else "TE-PAI=LOW"
        ax.set_title(f"seed {s}: TEPAI@{pkT:.1f}, Trot@{pkR:.1f}\n[{ok}]", fontsize=10, fontweight="bold")
        ax.set_xlabel("Frequency (rad/s)")
    for ax in axes[n:]:
        ax.set_visible(False)
    axes[0].set_ylabel("$I(E)$/max"); axes[0].legend(fontsize=8, loc="upper right")
    n_gap = sum(1 for r in results if near(r[5]))
    fig.suptitle(f"fig5 seed robustness (quick cond: N={f5.N_QUBITS}, n_t={N_T}, M={M}, "
                 f"depol p2={f5.NOISE.p2:.0e}) -- TE-PAI resolves gap in {n_gap}/{len(results)} seeds",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = os.path.join(HERE, "fig5_seed_robust.png")
    fig.savefig(out, dpi=160)
    print(f"[robust] saved {out}", flush=True)


if __name__ == "__main__":
    main()
