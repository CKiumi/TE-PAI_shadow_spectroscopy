"""Verify whether TE-PAI's stronger high-frequency gap peaks (g4=10.07, g5=11.71)
are REAL (Trotter under-resolves them) or an ARTIFACT of TE-PAI's variance.

Decisive test: compute the EXACT noiseless shadow-spectroscopy spectrum -- exact
time evolution e^{-iHt} (no Trotter error), exact observable expectations (no shadow
noise, M=inf) -- through the SAME post-processing (standardise -> Ljung-Box -> taper ->
SVD -> FFT). This is the ground-truth resolvability of each gap by shadow spectroscopy.

  * If the exact spectrum shows clear g4/g5 peaks  -> they are real; Trotter
    under-resolves and TE-PAI recovers them (a genuine TE-PAI advantage).
  * If the exact spectrum has g4/g5 at the floor   -> TE-PAI is inflating them
    (artifact of its Gamma^2 variance through the nonlinear SVD).

Same setup as fig5 (N=5 Heisenberg, dominant-ground init over LEVELS, k=3, t_max=6,
n_t=120, damping=0.05, pad=4). Compares against the cached TE-PAI / Trotter noiseless
spectra in ../inc/fig5_incremental_M*.npz.

Run:  uv run python paper/fig5/verify/verify_highfreq.py
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
INC = os.path.join(HERE, "..", "inc")

# --- fig5 settings (hardcoded to avoid import-path issues) ---
N, LEVELS, EPS, K_LOCAL = 5, [0, 1, 2, 4, 7, 9], 0.25, 3
T_MAX, N_T, DAMPING, PAD, CUTOFF, LJUNG = 6.0, 120, 0.05, 4, 4, True
GAPS = {"g1": 2.8831, "g2": 4.4755, "g3": 6.4755, "g4": 10.0664, "g5": 11.7115}
OFFS = [8.7, 13.2, 15.5]                      # off-gap high-freq floor checkpoints


def vmax(fr, y, f, w=0.5):
    m = (fr > f - w) & (fr < f + w)
    return float(y[m].max()) if m.any() else 0.0


def prominences(fr, y):
    floor = np.median([vmax(fr, y, f) for f in OFFS])
    return {g: vmax(fr, y, gf) / (floor or 1) for g, gf in GAPS.items()}, floor


def main():
    H = Heisenberg_Hamil(N, 1.0, 1.0, 1.0)
    E, V = H.eigh()
    levels, idx = np.unique(np.round(E, 6), return_index=True)
    sel, sel_E = idx[LEVELS], levels[LEVELS]
    w = np.ones(len(LEVELS)); w[1:] = EPS
    psi = (V[:, sel] * w).sum(axis=1); psi /= np.linalg.norm(psi)
    c0 = V.conj().T @ psi                                       # init in energy basis

    paulis = k_local_paulis(N, K_LOCAL)
    obs = make_observables(paulis, N)
    times = np.linspace(0, T_MAX, N_T); dt = float(times[1] - times[0])
    st = QuantumState(N)
    Dex = np.zeros((N_T, len(paulis)))
    for ti, t in enumerate(times):
        st.load(V @ (np.exp(-1j * E * t) * c0))                 # exact e^{-iHt}|psi>
        for j, ob in enumerate(obs):
            Dex[ti, j] = 1.0 if ob is None else ob.get_expectation_value(st).real
    frx, spx = Spectroscopy(dt, CUTOFF, DAMPING, PAD).spectrum(Dex, LJUNG)
    yx = np.abs(spx)
    promx, floorx = prominences(frx, yx)
    print(f"[verify] EXACT noiseless spectrum  off-gap floor={floorx:.4f}")
    print("         " + "  ".join(f"{g}@{GAPS[g]:.1f} prom={promx[g]:.1f}" for g in GAPS))

    # cached TE-PAI / Trotter noiseless (M=10000) for comparison
    d = np.load(os.path.join(INC, "fig5_incremental_M10000.npz"))
    rows = {"EXACT": (frx, yx, promx, floorx)}
    for key, lab in [("TE-PAI", "TE-PAI"), ("Trotter_fine", "Trotter")]:
        fr = d[key + "_f"]; y = np.abs(d[key + "_s"]); pr, fl = prominences(fr, y)
        rows[lab] = (fr, y, pr, fl)
        print(f"[verify] {lab:8s} (M=10000)  floor={fl:.4f}  " +
              "  ".join(f"{g} prom={pr[g]:.1f}" for g in ["g3", "g4", "g5"]))

    # verdict
    print("\n[verify] g4/g5 prominence:  EXACT vs TE-PAI vs Trotter")
    for g in ["g4", "g5"]:
        ex, tp, tr = promx[g], rows["TE-PAI"][2][g], rows["Trotter"][2][g]
        verdict = ("REAL: exact resolves it, Trotter under-resolves" if ex > 2 and tr < 1.8
                   else "ARTIFACT: exact floor, TE-PAI inflates" if ex < 1.8 and tp > 2
                   else "mixed/unclear")
        print(f"   {g}: exact={ex:.1f}  TE-PAI={tp:.1f}  Trotter={tr:.1f}  -> {verdict}")

    _plot(rows)


def _plot(rows):
    sty = {"EXACT": dict(color="black", ls="-", lw=2.0),
           "TE-PAI": dict(color="red", ls="-", lw=1.6),
           "Trotter": dict(color="blue", ls="-", lw=1.6)}
    fig, ax = plt.subplots(figsize=(11, 6))
    for lab in ["EXACT", "TE-PAI", "Trotter"]:
        fr, y, pr, fl = rows[lab]
        ax.plot(fr, y / (y.max() or 1), label=f"{lab} (g4 prom {pr['g4']:.1f}, g5 {pr['g5']:.1f})", **sty[lab])
    for j, (g, gf) in enumerate(GAPS.items()):
        ax.axvline(gf, color="0.6", ls="--", lw=1.0, label="gaps" if j == 0 else None)
    ax.set_xlim(0, 13.5); ax.set_ylim(0, 1.05); ax.grid(True, ls="--", lw=0.5, alpha=0.5)
    ax.set_xlabel("Frequency (rad/s)"); ax.set_ylabel("$I(E)$/max")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_title("High-freq gap verification: EXACT noiseless vs TE-PAI vs Trotter (N=5, noiseless)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    out = os.path.join(HERE, "verify_highfreq.png")
    fig.savefig(out, dpi=170); print(f"\n[verify] saved {out}")


if __name__ == "__main__":
    main()
