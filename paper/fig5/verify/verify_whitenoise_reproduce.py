"""Direct test: does adding white noise to clean (Trotter/exact) data reproduce the
TE-PAI high-frequency behaviour seen in verify_highfreq.png?

Same plot as verify_highfreq.png but with one extra curve: the clean exact data matrix
(= noiseless Trotter limit, no shadow noise) PLUS injected white Gaussian noise of std
sigma comparable to TE-PAI's per-entry noise. If that "clean + white noise" spectrum
reproduces TE-PAI's g4/g5 peaks, the high-freq surfacing is a generic white-noise effect,
not a TE-PAI property.

Run:  uv run python paper/fig5/verify/verify_whitenoise_reproduce.py [sigma]
"""
from __future__ import annotations
import os, sys
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
SIGMA = float(sys.argv[1]) if len(sys.argv) > 1 else 0.2
GAPS = {"g1": 2.8831, "g2": 4.4755, "g3": 6.4755, "g4": 10.0664, "g5": 11.7115}
OFFS = [8.7, 13.2, 15.5]


def vmax(fr, y, f, w=0.5):
    m = (fr > f - w) & (fr < f + w); return float(y[m].max()) if m.any() else 0.0


def proms(fr, y):
    fl = np.median([vmax(fr, y, f) for f in OFFS]); return {g: vmax(fr, y, gf) / (fl or 1) for g, gf in GAPS.items()}


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
    frx, spx = spec.spectrum(D, LJUNG)                                  # exact clean
    rng = np.random.default_rng(2)
    frw, spw = spec.spectrum(D + rng.standard_normal(D.shape) * SIGMA, LJUNG)   # clean + white noise

    d = np.load(os.path.join(INC, "fig5_incremental_M10000.npz"))       # cached TE-PAI / Trotter
    frT, spT = d["TE-PAI_f"], np.abs(d["TE-PAI_s"])
    frR, spR = d["Trotter_fine_f"], np.abs(d["Trotter_fine_s"])

    rows = [("EXACT (clean)", frx, np.abs(spx), "black", "-", 2.0),
            ("Trotter (clean, M=1e4)", frR, spR, "blue", "-", 1.5),
            ("TE-PAI (M=1e4)", frT, spT, "red", "-", 1.7),
            (f"clean + white noise $\\sigma$={SIGMA}", frw, np.abs(spw), "green", "--", 2.0)]
    print(f"sigma={SIGMA}")
    for lab, fr, y, *_ in rows:
        pr = proms(fr, y); print(f"  {lab:28s} g3={pr['g3']:.1f} g4={pr['g4']:.1f} g5={pr['g5']:.1f}")

    fig, ax = plt.subplots(figsize=(11, 6))
    for lab, fr, y, c, ls, lw in rows:
        ax.plot(fr, y / (y.max() or 1), color=c, ls=ls, lw=lw, label=lab)
    for j, gf in enumerate(GAPS.values()):
        ax.axvline(gf, color="0.6", ls=":", lw=1.0, label="gaps" if j == 0 else None)
    ax.set_xlim(0, 13.5); ax.set_ylim(0, 1.05); ax.grid(True, ls="--", lw=0.5, alpha=0.5)
    ax.set_xlabel("Frequency (rad/s)"); ax.set_ylabel("$I(E)$/max"); ax.legend(fontsize=9, loc="upper right")
    ax.set_title(f"Does clean+white-noise (green) reproduce TE-PAI (red)? (N=5, noiseless, cutoff=4)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(); out = os.path.join(HERE, f"verify_whitenoise_reproduce_s{SIGMA}.png")
    fig.savefig(out, dpi=170); print(f"saved {out}")


if __name__ == "__main__":
    main()
