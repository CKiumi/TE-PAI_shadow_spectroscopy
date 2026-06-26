"""Grid: clean exact D + white noise at many sigma, each overlaid with TE-PAI (red) and
exact (black), to see whether ANY sigma reproduces TE-PAI's clean moderate high-freq
surfacing. (White noise is injected independently per entry -- the wrong correlation
structure for TE-PAI, which is exactly the point being tested.)

Run:  uv run python paper/fig5/verify/verify_sigma_grid.py
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt
from qulacs import QuantumState
from pai_shadow.hamil import Heisenberg_Hamil
from pai_shadow.shadow_spectro import Spectroscopy, k_local_paulis
from pai_shadow.circuit import make_observables

np.seterr(all="ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); INC = os.path.join(HERE, "..", "inc")
N, LEVELS, EPS, K_LOCAL = 5, [0, 1, 2, 4, 7, 9], 0.25, 3
T_MAX, N_T, DAMPING, PAD, CUTOFF, LJUNG = 6.0, 120, 0.05, 4, 4, True
GAPS = [2.8831, 4.4755, 6.4755, 10.0664, 11.7115]
OFFS = [8.7, 13.2, 15.5]
SIGMAS = [0.03, 0.06, 0.10, 0.15, 0.20, 0.30, 0.50, 0.80]


def vmax(fr, y, f, w=0.5):
    m = (fr > f - w) & (fr < f + w); return float(y[m].max()) if m.any() else 0.0


def main():
    H = Heisenberg_Hamil(N, 1.0, 1.0, 1.0); E, V = H.eigh()
    levels, idx = np.unique(np.round(E, 6), return_index=True)
    sel = idx[LEVELS]; w = np.ones(len(LEVELS)); w[1:] = EPS
    psi = (V[:, sel] * w).sum(axis=1); psi /= np.linalg.norm(psi); c0 = V.conj().T @ psi
    paulis = k_local_paulis(N, K_LOCAL); obs = make_observables(paulis, N)
    times = np.linspace(0, T_MAX, N_T); dt = float(times[1] - times[0])
    st = QuantumState(N); D = np.zeros((N_T, len(paulis)))
    for ti, t in enumerate(times):
        st.load(V @ (np.exp(-1j * E * t) * c0))
        for j, ob in enumerate(obs):
            D[ti, j] = 1.0 if ob is None else ob.get_expectation_value(st).real
    spec = Spectroscopy(dt, CUTOFF, DAMPING, PAD)
    frx, spx = spec.spectrum(D, LJUNG); yx = np.abs(spx); yx = yx / yx.max()
    d = np.load(os.path.join(INC, "fig5_incremental_M10000.npz"))
    frT, yT = d["TE-PAI_f"], np.abs(d["TE-PAI_s"]); yT = yT / yT.max()

    fig, axes = plt.subplots(2, 4, figsize=(20, 9), sharex=True, sharey=True); axes = axes.ravel()
    rng = np.random.default_rng(2)
    for ax, s in zip(axes, SIGMAS):
        fr, sp = spec.spectrum(D + rng.standard_normal(D.shape) * s, LJUNG)
        yw = np.abs(sp); fl = np.median([vmax(fr, yw, f) for f in OFFS])
        g4p = vmax(fr, yw, GAPS[3]) / (fl or 1)
        yw = yw / yw.max()
        ax.plot(frx, yx, color="black", lw=1.6, label="EXACT")
        ax.plot(frT, yT, color="red", lw=1.6, label="TE-PAI")
        ax.plot(fr, yw, color="green", ls="--", lw=1.8, label=f"white $\\sigma$={s}")
        for j, g in enumerate(GAPS):
            ax.axvline(g, color="0.6", ls=":", lw=0.9, label="gaps" if j == 0 else None)
        ax.set_xlim(0, 13.5); ax.set_ylim(0, 1.05); ax.grid(True, ls="--", lw=0.5, alpha=0.5)
        ax.set_title(f"$\\sigma$={s}  (white g4 prom={g4p:.1f})", fontsize=10, fontweight="bold")
        ax.set_xlabel("Frequency (rad/s)"); ax.legend(fontsize=7, loc="upper right")
    axes[0].set_ylabel("$I(E)$/max"); axes[4].set_ylabel("$I(E)$/max")
    fig.suptitle("Does any white-noise sigma reproduce TE-PAI (red)? EXACT(black) + TE-PAI(red) + white(green)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(); out = os.path.join(HERE, "verify_sigma_grid.png")
    fig.savefig(out, dpi=140); print(f"saved {out}")


if __name__ == "__main__":
    main()
