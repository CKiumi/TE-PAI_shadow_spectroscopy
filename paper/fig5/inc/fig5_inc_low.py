"""Low-M incremental: M = 1000, 2000, 3000 (batches [1000,1000,1000]) for the N=5
fig5 TE-PAI-vs-Trotter comparison. Self-contained (settings hardcoded to match
fig5_trotter_vs_tepai.py N=5). This time the cumulative DATA MATRICES D are saved too,
so the spectrum can be recomputed at any cutoff / pad / damping for free later.

Run:  uv run python paper/fig5/inc/fig5_inc_low.py
"""
from __future__ import annotations
import os, time
import numpy as np
import matplotlib.pyplot as plt
from pai_shadow.hamil import Heisenberg_Hamil
from pai_shadow.circuit import NoiseSpec
from pai_shadow.shadow_spectro import (Spectroscopy, dominant_gap,
                                       te_pai_shadow_spectroscopy, trotter_shadow_spectroscopy)

HERE = os.path.dirname(os.path.abspath(__file__))
N, LEVELS, EPS, K_LOCAL = 5, [0, 1, 2, 4, 7, 9], 0.25, 3
DELTA, N_DIV = np.pi / 2**6, 700
T_MAX, N_T, DAMPING, PAD, CUTOFF, LJUNG = 6.0, 120, 0.05, 4, 4, True
NOISE = NoiseSpec(p1=4e-5, p2=4e-4, kind="depolarizing")
BATCHES = [1000, 1000, 1000]; CUM = list(np.cumsum(BATCHES))
NJ = os.cpu_count()
SEED_BASE = {"TE-PAI": 100, "TE-PAI noisy": 200, "Trotter fine": 300, "Trotter fine noisy": 400}


def main():
    H = Heisenberg_Hamil(N, 1.0, 1.0, 1.0); E, V = H.eigh()
    levels, idx = np.unique(np.round(E, 6), return_index=True)
    sel, sel_E = idx[LEVELS], levels[LEVELS]
    w = np.ones(len(LEVELS)); w[1:] = EPS
    psi = (V[:, sel] * w).sum(axis=1); psi /= np.linalg.norm(psi)
    dom = [round(float(sel_E[k] - sel_E[0]), 4) for k in range(1, len(LEVELS))]
    cross = sorted({round(float(sel_E[j] - sel_E[i]), 4) for i in range(1, len(LEVELS)) for j in range(i + 1, len(LEVELS))})
    times = np.linspace(0, T_MAX, N_T); DT = T_MAX / N_DIV; dt = float(times[1] - times[0])
    print(f"[low] N={N} gaps={dom} batches={BATCHES} -> cum={CUM}", flush=True)

    def tepai_D(M, noise, seed):
        return te_pai_shadow_spectroscopy(H, psi, times, delta=DELTA, M=M, n_shots=1, trotter_step=DT,
            n_trotter_max=None, k=K_LOCAL, noise=noise, damping=DAMPING, pad=PAD, seed=seed, n_jobs=NJ,
            label="tepai", return_data=True)[2]

    def trotter_D(M, noise, seed):
        return trotter_shadow_spectroscopy(H, psi, times, trotter_step=DT, shadow_size=M, k=K_LOCAL,
            noise=noise, density=(noise is not None), damping=DAMPING, pad=PAD, seed=seed, n_jobs=NJ,
            label="trot", return_data=True)[2]

    streams = {"TE-PAI": (tepai_D, None), "TE-PAI noisy": (tepai_D, NOISE),
               "Trotter fine": (trotter_D, None), "Trotter fine noisy": (trotter_D, NOISE)}
    cumD = {name: {} for name in streams}
    for name, (fn, noise) in streams.items():
        Dsum, Msum = None, 0
        for bi, b in enumerate(BATCHES):
            t0 = time.perf_counter()
            D = fn(b, noise, SEED_BASE[name] + bi)
            Dsum = b * D if Dsum is None else Dsum + b * D
            Msum += b
            cumD[name][Msum] = Dsum / Msum
            print(f"[low] {name}: +{b} -> M={Msum} ({time.perf_counter()-t0:.0f}s)", flush=True)

    key = lambda s: s.replace(" ", "_")
    for cum in CUM:
        spec = {n: Spectroscopy(dt, CUTOFF, DAMPING, PAD).spectrum(cumD[n][cum], LJUNG) for n in streams}
        np.savez(os.path.join(HERE, f"fig5_incremental_M{cum}.npz"), dt=dt, M=cum,
                 dom=np.array(dom), cross=np.array(cross),
                 **{key(n) + "_f": s[0] for n, s in spec.items()},
                 **{key(n) + "_s": s[1] for n, s in spec.items()},
                 **{key(n) + "_D": cumD[n][cum] for n in streams})       # D saved this time
        _plot(spec, dom, cross, dt, cum)
    print("[low] done (D matrices saved -> future cutoff/pad/damping are free)", flush=True)


def _plot(spec, dom, cross, dt, cum):
    sty = {"TE-PAI": dict(color="red", lw=1.7), "Trotter fine": dict(color="blue", lw=1.5)}
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True, sharey=True)
    for ax, (t, suf) in zip(axes, [("noiseless", ""), ("with gate noise", " noisy")]):
        for base in ["TE-PAI", "Trotter fine"]:
            fr, sp = spec[base + suf]; y = np.abs(sp)
            ax.plot(fr, y / (y.max() or 1), label=f"{base} (peak@{dominant_gap(fr, sp):.1f})", **sty[base])
        for j, g in enumerate(dom): ax.axvline(g, color="black", ls="--", lw=1.0, label="gaps" if j == 0 else None)
        for g in cross: ax.axvline(g, color="0.75", ls=":", lw=0.8)
        ax.set_xlim(0, max(dom + cross) + 1); ax.set_ylim(0, 1.05); ax.grid(True, ls="--", lw=0.5, alpha=0.5)
        ax.set_xlabel("Frequency (rad/s)"); ax.set_title(t, fontsize=12, fontweight="bold"); ax.legend(fontsize=9)
    axes[0].set_ylabel("$I(E)$/max")
    fig.suptitle(f"Fig.5 incremental -- M={cum} (N=5, depol p2=4e-4)", fontsize=13, fontweight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, f"fig5_incremental_M{cum}.png"), dpi=160)


if __name__ == "__main__":
    main()
