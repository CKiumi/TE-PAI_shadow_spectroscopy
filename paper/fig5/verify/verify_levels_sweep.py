"""Same EXACT/TE-PAI/Trotter spectrum comparison across several initial states (LEVELS).

For each choice of superposed eigenlevels (dominant ground + small excited), we compute
the noiseless shadow spectrum three ways -- exact (M=inf, no Trotter error), TE-PAI, and
Trotter -- through identical post-processing (cutoff=4), and overlay them. This checks
whether the high-frequency behaviour (TE-PAI showing peaks that exact/Trotter leave at
the floor) is consistent across initial states or specific to one setup.

Run:  uv run python paper/fig5/verify/verify_levels_sweep.py
"""
from __future__ import annotations
import os, time
import numpy as np
import matplotlib.pyplot as plt
from qulacs import QuantumState
from pai_shadow.hamil import Heisenberg_Hamil
from pai_shadow.shadow_spectro import (Spectroscopy, k_local_paulis,
                                       te_pai_shadow_spectroscopy, trotter_shadow_spectroscopy)
from pai_shadow.circuit import make_observables

np.seterr(all="ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
N, EPS, K_LOCAL = 5, 0.25, 3
DELTA, N_DIV = np.pi / 2**6, 700
T_MAX, N_T, DAMPING, PAD, CUTOFF, LJUNG = 6.0, 120, 0.05, 4, 4, True
M, NJ = 4000, os.cpu_count()
CONFIGS = [[0, 1, 2, 3, 4, 5], [0, 1, 2, 4, 7, 9], [0, 2, 4, 6, 8], [0, 1, 3, 6, 9]]


def main():
    H = Heisenberg_Hamil(N, 1.0, 1.0, 1.0); E, V = H.eigh()
    levels, idx = np.unique(np.round(E, 6), return_index=True)
    paulis = k_local_paulis(N, K_LOCAL); obs = make_observables(paulis, N)
    times = np.linspace(0, T_MAX, N_T); DT = T_MAX / N_DIV; dt = float(times[1] - times[0])
    st = QuantumState(N)
    results = []
    for cfg in CONFIGS:
        t0 = time.perf_counter()
        sel, sel_E = idx[cfg], levels[cfg]
        w = np.ones(len(cfg)); w[1:] = EPS
        psi = (V[:, sel] * w).sum(axis=1); psi /= np.linalg.norm(psi)
        dom = sorted(round(float(sel_E[k] - sel_E[0]), 3) for k in range(1, len(cfg)))
        # exact
        c0 = V.conj().T @ psi; D = np.zeros((N_T, len(paulis)))
        for ti, t in enumerate(times):
            st.load(V @ (np.exp(-1j * E * t) * c0))
            for j, ob in enumerate(obs):
                D[ti, j] = 1.0 if ob is None else ob.get_expectation_value(st).real
        frx, spx = Spectroscopy(dt, CUTOFF, DAMPING, PAD).spectrum(D, LJUNG)
        # TE-PAI & Trotter (noiseless)
        frT, spT = te_pai_shadow_spectroscopy(H, psi, times, delta=DELTA, M=M, n_shots=1,
            trotter_step=DT, n_trotter_max=None, k=K_LOCAL, noise=None, damping=DAMPING,
            pad=PAD, seed=2, n_jobs=NJ, label=f"TEPAI {cfg}")
        frR, spR = trotter_shadow_spectroscopy(H, psi, times, trotter_step=DT, shadow_size=M,
            k=K_LOCAL, noise=None, density=False, damping=DAMPING, pad=PAD, seed=1, n_jobs=NJ,
            label=f"Trot {cfg}")
        results.append((cfg, dom, (frx, np.abs(spx)), (frT, np.abs(spT)), (frR, np.abs(spR))))
        print(f"[levels] {cfg} dom={dom} done ({time.perf_counter()-t0:.0f}s)", flush=True)
    _plot(results, dt)


def _plot(results, dt):
    n = len(results); fig, axes = plt.subplots(2, 2, figsize=(15, 9)); axes = axes.ravel()
    for ax, (cfg, dom, ex, tp, tr) in zip(axes, results):
        for (fr, y), lab, c, lw in [(ex, "EXACT", "black", 2.0), (tp, "TE-PAI", "red", 1.5), (tr, "Trotter", "blue", 1.5)]:
            ax.plot(fr, y / (y.max() or 1), color=c, lw=lw, label=lab)
        for j, g in enumerate(dom):
            ax.axvline(g, color="0.6", ls="--", lw=1.0, label="gaps" if j == 0 else None)
        ax.set_xlim(0, max(dom) + 1.5); ax.set_ylim(0, 1.05); ax.grid(True, ls="--", lw=0.5, alpha=0.5)
        ax.set_xlabel("Frequency (rad/s)"); ax.set_ylabel("$I(E)$/max")
        ax.set_title(f"init levels {cfg}\n gaps={dom}", fontsize=10, fontweight="bold")
        ax.legend(fontsize=8, loc="upper right")
    fig.suptitle(f"EXACT vs TE-PAI vs Trotter across initial states (N={N}, noiseless, M={M}, cutoff={CUTOFF})",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(); out = os.path.join(HERE, "verify_levels_sweep.png")
    fig.savefig(out, dpi=160); print(f"[levels] saved {out}")


if __name__ == "__main__":
    main()
