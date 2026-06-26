"""Does the EXACT spectrum's failure to show g4/g5 come from the SVD truncation
(cutoff = number of retained singular vectors), or is it fundamental?

We build the exact noiseless data matrix once and re-post-process it at several cutoff
values, tracking the g4/g5 prominence. If raising cutoff makes g4/g5 appear, the
suppression is a method (truncation) limitation -- and the question of whether TE-PAI
legitimately surfaces them reopens. If g4/g5 stay at the floor for all cutoffs, they
are genuinely unresolved by shadow spectroscopy and TE-PAI's peaks are artifacts.

Run:  uv run python paper/fig5/verify/verify_cutoff.py
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
HERE = os.path.dirname(os.path.abspath(__file__))
N, LEVELS, EPS, K_LOCAL = 5, [0, 1, 2, 4, 7, 9], 0.25, 3
T_MAX, N_T, DAMPING, PAD, LJUNG = 6.0, 120, 0.05, 4, True
GAPS = {"g1": 2.8831, "g2": 4.4755, "g3": 6.4755, "g4": 10.0664, "g5": 11.7115}
OFFS = [8.7, 13.2, 15.5]
CUTOFFS = [2, 4, 6, 8, 10, 14]


def vmax(fr, y, f, w=0.5):
    m = (fr > f - w) & (fr < f + w); return float(y[m].max()) if m.any() else 0.0


def main():
    H = Heisenberg_Hamil(N, 1.0, 1.0, 1.0); E, V = H.eigh()
    levels, idx = np.unique(np.round(E, 6), return_index=True)
    sel = idx[LEVELS]; w = np.ones(len(LEVELS)); w[1:] = EPS
    psi = (V[:, sel] * w).sum(axis=1); psi /= np.linalg.norm(psi)
    c0 = V.conj().T @ psi
    paulis = k_local_paulis(N, K_LOCAL); obs = make_observables(paulis, N)
    times = np.linspace(0, T_MAX, N_T); dt = float(times[1] - times[0])
    st = QuantumState(N); D = np.zeros((N_T, len(paulis)))
    for ti, t in enumerate(times):
        st.load(V @ (np.exp(-1j * E * t) * c0))
        for j, ob in enumerate(obs):
            D[ti, j] = 1.0 if ob is None else ob.get_expectation_value(st).real

    print("cutoff  g1    g2    g3    g4    g5     (prominence vs off-gap floor)")
    specs = {}
    for c in CUTOFFS:
        fr, sp = Spectroscopy(dt, c, DAMPING, PAD).spectrum(D, LJUNG)
        y = np.abs(sp); floor = np.median([vmax(fr, y, f) for f in OFFS])
        pr = {g: vmax(fr, y, gf) / (floor or 1) for g, gf in GAPS.items()}
        specs[c] = (fr, y)
        print(f"  {c:3d}  " + "  ".join(f"{pr[g]:4.1f}" for g in GAPS))

    fig, ax = plt.subplots(figsize=(11, 6))
    for c in CUTOFFS:
        fr, y = specs[c]; ax.plot(fr, y / (y.max() or 1), lw=1.4, label=f"cutoff={c}")
    for j, gf in enumerate(GAPS.values()):
        ax.axvline(gf, color="0.6", ls="--", lw=1.0, label="gaps" if j == 0 else None)
    ax.set_xlim(0, 13.5); ax.set_ylim(0, 1.05); ax.grid(True, ls="--", lw=0.5, alpha=0.5)
    ax.set_xlabel("Frequency (rad/s)"); ax.set_ylabel("$I(E)$/max")
    ax.legend(fontsize=9); ax.set_title("EXACT noiseless spectrum vs SVD cutoff (does g4/g5 appear?)",
                                        fontsize=12, fontweight="bold")
    fig.tight_layout(); out = os.path.join(HERE, "verify_cutoff.png")
    fig.savefig(out, dpi=170); print(f"saved {out}")


if __name__ == "__main__":
    main()
