"""Fig. 5 comparison, computed INCREMENTALLY in shot batches (no wasted work).

The shadow estimator for each observable is a plain shot-MEAN (a weighted mean for
TE-PAI's PAI signs) -- see ClassicalShadow.expectation, ``vals.mean()``. Therefore the
shot-weighted average of per-batch data matrices is EXACTLY the mean over all shots:

    D(M_total) = sum_i M_i D_i / sum_i M_i  =  (sum over all shots) / M_total

i.e. accumulating the running weighted sum and dividing gives the SAME D (and hence the
same spectrum) as one run at M_total -- it is NOT an approximation, and no per-shot data
needs to be stored. The post-processing (standardise -> Ljung -> taper -> FFT) is cheap,
so we re-run only that at each cumulative total.

Batches [4000, 2000, 2000, 2000] give the spectra at M = 4000, 6000, 8000, 10000 for a
TOTAL cost of 10000 shots/stream -- versus 4000+6000+8000+10000 = 28000 if each M were
run from scratch.

All physics/noise settings are imported from fig5_trotter_vs_tepai.py (N=5, dominant-
ground state, Delta=pi/2**6, depolarizing p2=4e-4, ...). Streams: TE-PAI and fine
Trotter, each noiseless and noisy.

Run:  uv run python paper/fig5_incremental.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

try:
    import paper.fig5_trotter_vs_tepai as f5
except ModuleNotFoundError:
    import fig5_trotter_vs_tepai as f5
from pai_shadow.hamil import Heisenberg_Hamil
from pai_shadow.shadow_spectro import (
    Spectroscopy,
    dominant_gap,
    te_pai_shadow_spectroscopy,
    trotter_shadow_spectroscopy,
)

_TEST = len(sys.argv) > 1 and sys.argv[1] == "test"
BATCHES = [200, 100] if _TEST else [4000, 2000, 2000, 2000]   # incremental shot batches
CUM = list(np.cumsum(BATCHES))                # paper: -> 4000, 6000, 8000, 10000
T_MAX, N_T = (4.0, 40) if _TEST else (6.0, 120)          # paper grid
CUTOFF, LJUNG = 4, True
NJ = os.cpu_count()
# distinct RNG base per stream so every batch draws independent shots
SEED_BASE = {"TE-PAI": 100, "TE-PAI noisy": 200, "Trotter fine": 300, "Trotter fine noisy": 400}


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
    print(f"[inc] N={f5.N_QUBITS} levels={f5.LEVELS} gaps={dom} batches={BATCHES} -> cum={CUM}", flush=True)

    def tepai_D(M, noise, seed):
        return te_pai_shadow_spectroscopy(
            H, psi, times, delta=f5.DELTA, M=M, n_shots=1, trotter_step=DT, n_trotter_max=None,
            k=f5.K_LOCAL, noise=noise, damping=f5.DAMPING, pad=f5.PAD, seed=seed, n_jobs=NJ,
            label="tepai", return_data=True)[2]

    def trotter_D(M, noise, seed):
        return trotter_shadow_spectroscopy(
            H, psi, times, trotter_step=DT, shadow_size=M, k=f5.K_LOCAL, noise=noise,
            density=(noise is not None), damping=f5.DAMPING, pad=f5.PAD, seed=seed, n_jobs=NJ,
            label="trot", return_data=True)[2]

    streams = {
        "TE-PAI": (tepai_D, None), "TE-PAI noisy": (tepai_D, f5.NOISE),
        "Trotter fine": (trotter_D, None), "Trotter fine noisy": (trotter_D, f5.NOISE),
    }

    # accumulate weighted D per stream, snapshot at each cumulative total
    cumD = {name: {} for name in streams}
    for name, (fn, noise) in streams.items():
        Dsum, Msum = None, 0
        for bi, b in enumerate(BATCHES):
            t0 = time.perf_counter()
            D = fn(b, noise, SEED_BASE[name] + bi)
            Dsum = b * D if Dsum is None else Dsum + b * D
            Msum += b
            cumD[name][Msum] = Dsum / Msum
            print(f"[inc] {name}: +{b} shots -> M={Msum} ({time.perf_counter()-t0:.0f}s)", flush=True)

    # for each cumulative M, post-process and plot
    for cum in CUM:
        spec = {name: Spectroscopy(dt, CUTOFF, f5.DAMPING, f5.PAD).spectrum(cumD[name][cum], LJUNG)
                for name in streams}
        np.savez(f"paper/fig5_incremental_M{cum}.npz", dt=dt, M=cum, dom=np.array(dom), cross=np.array(cross),
                 **{n.replace(" ", "_") + "_f": s[0] for n, s in spec.items()},
                 **{n.replace(" ", "_") + "_s": s[1] for n, s in spec.items()})
        _plot(spec, dom, cross, dt, cum)


def _plot(spec, dom, cross, dt, cum) -> None:
    sty = {"TE-PAI": dict(color="red", ls="-", lw=1.9), "Trotter fine": dict(color="blue", ls="-", lw=1.6)}
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True, sharey=True)
    for ax, (ptitle, suf) in zip(axes, [("noiseless", ""), ("with gate noise", " noisy")]):
        for base in ["TE-PAI", "Trotter fine"]:
            fr, sp = spec[base + suf]; y = np.abs(sp)
            pk = dominant_gap(fr, sp)
            ax.plot(fr, y / (y.max() or 1.0), label=f"{base} (peak@{pk:.1f})", **sty[base])
        for j, g in enumerate(dom):
            ax.axvline(g, color="black", ls="--", lw=1.1, label="dominant gaps" if j == 0 else None)
        for g in cross:
            ax.axvline(g, color="0.7", ls=":", lw=0.9)
        ax.grid(True, ls="--", lw=0.5, alpha=0.5)
        ax.set_xlim(0, max(dom + cross) + 1.0); ax.set_ylim(0, 1.05)
        ax.set_xlabel("Frequency (rad/s)"); ax.set_title(ptitle, fontsize=13, fontweight="bold")
        ax.legend(fontsize=9, loc="upper right")
    axes[0].set_ylabel("$I(E)$/max")
    fig.suptitle(f"Fig. 5 incremental -- M={cum} (batched, {f5.N_QUBITS}-qubit, "
                 f"{f5.NOISE.kind} p2={f5.NOISE.p2:.0e})", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = f"paper/fig5_incremental_M{cum}.png"
    fig.savefig(out, dpi=170)
    print(f"[inc] saved {out}", flush=True)


if __name__ == "__main__":
    main()
