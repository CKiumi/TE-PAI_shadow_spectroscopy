"""Reproduce Fig. 1: energy spectra of the 10-qubit Heisenberg model.

Paper: "Low-Resource Quantum Energy Gap Estimation via Randomization".

Fig. 1 compares the energy spectrum recovered by **TE-PAI shadow spectroscopy**
(solid lines, three sample/shot splits of a fixed budget) against standard
**Trotter-based shadow spectroscopy** (dotted line), for a noise-free 10-qubit
1D Heisenberg model (Eq. 9):

    H = sum_i  Jx Xi Xi+1 + Jy Yi Yi+1 + Jz Zi Zi+1 ,   Jx = Jy = Jz = 1.

The initial state is prepared as (|E_0> + |E_10>)/sqrt(2), a superposition of the
ground state and the 10th excited state, so the dominant spectral peak sits at
the transition gap dE_{0,10} = E_10 - E_0 ~= 4.36.

Both methods estimate every 3-local Pauli observable from classical-shadow
snapshots at N_t time points, keep the top 10% most autocorrelated signals
(Ljung-Box), and Fourier-analyse the result; peaks appear at the energy gaps.

Two presets (set PRESET below or pass "paper" / "quick" on the command line):

* "paper"  -- the exact parameters from the paper (N_t=90, K=650, Delta=pi/2**7,
              1000 executions/time point). Faithful but heavy: hours on a laptop.
* "quick"  -- a reduced-budget run (default) that still resolves the dE_{0,10}
              peak in a few minutes, to validate the pipeline.

Run:  uv run python paper/fig1_heisenberg_spectrum.py [paper|quick]
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

from pai_shadow.hamil import Heisenberg_Hamil
from pai_shadow.shadow_spectro import (
    dominant_gap,
    te_pai_shadow_spectroscopy,
    trotter_shadow_spectroscopy,
)

# --------------------------------------------------------------------------- #
#  Parameters                                                                  #
# --------------------------------------------------------------------------- #
N_QUBITS = 10                      # 10-qubit Heisenberg chain (paper Fig. 1)
EXCITED = 10                       # initial state = (|E_0> + |E_10>)/sqrt(2)
K_LOCAL = 3                        # estimate all 3-local Pauli observables

PRESETS = {
    # name:   dict(n_t, dt, delta, trotter_steps, configs=[(M_TEPAI, N_s)...],
    #              trotter_shots)
    "paper": dict(
        n_t=90, dt=0.11, delta=np.pi / 2**7, trotter_steps=650,
        configs=[(1000, 1), (500, 2), (250, 4)], trotter_shots=1000,
    ),
    "quick": dict(
        n_t=45, dt=0.11, delta=np.pi / 2**7, trotter_steps=150,
        configs=[(200, 1), (100, 2), (50, 4)], trotter_shots=200,
    ),
}

SEED = 0
N_JOBS = os.cpu_count()            # parallelise time points over all CPU cores


def main(preset: str = "quick") -> None:
    cfg = PRESETS[preset]
    print(f"[fig1] preset={preset!r}  n_jobs={N_JOBS}  params={cfg}")

    # 10-qubit Heisenberg model and the ground + 10th-excited superposition.
    H = Heisenberg_Hamil(N_QUBITS, 1.0, 1.0, 1.0)          # open chain
    e0, e_exc, g, x = H.get_ground_and_excited_state(n=EXCITED)
    init = (g + x) / np.sqrt(2.0)
    gap = e_exc - e0
    print(f"[fig1] E_0={e0:.4f}  E_{EXCITED}={e_exc:.4f}  "
          f"target gap dE_0,{EXCITED} = {gap:.4f}")

    times = np.arange(cfg["n_t"]) * cfg["dt"]

    # TE-PAI spectra: three (M_TE-PAI, N_s) splits of the same execution budget.
    tepai_spectra = []
    for j, (M, n_s) in enumerate(cfg["configs"]):
        t0 = time.perf_counter()
        freqs, spec = te_pai_shadow_spectroscopy(
            H, init, times, delta=cfg["delta"], M=M, n_shots=n_s,
            k=K_LOCAL, seed=SEED + j, n_jobs=N_JOBS,
        )
        peak = dominant_gap(freqs, spec)
        print(f"[fig1] TE-PAI (M={M}, N_s={n_s}): peak={peak:.3f}  "
              f"({time.perf_counter() - t0:.1f}s)")
        tepai_spectra.append((M, n_s, freqs, spec))

    # Trotter-based spectrum (single deep circuit, N_s shadow snapshots).
    t0 = time.perf_counter()
    ft, st = trotter_shadow_spectroscopy(
        H, init, times, n_steps=cfg["trotter_steps"],
        shadow_size=cfg["trotter_shots"], k=K_LOCAL, seed=SEED + 99, n_jobs=N_JOBS,
    )
    print(f"[fig1] Trotter (N_s={cfg['trotter_shots']}): "
          f"peak={dominant_gap(ft, st):.3f}  ({time.perf_counter() - t0:.1f}s)")

    _plot(tepai_spectra, (ft, st), gap, cfg, preset)


def _plot(tepai_spectra, trotter, gap, cfg, preset) -> None:
    ft, st = trotter
    colors = ["#d62728", "#2ca02c", "#ff7f0e"]
    fig, ax = plt.subplots(figsize=(6, 4))

    # Common normalisation (arbitrary units): all curves share one scale so the
    # relative peak intensities are meaningful (the paper shows Trotter highest).
    scale = max([np.max(np.abs(st))] + [np.max(np.abs(sp)) for *_, sp in tepai_spectra])
    scale = scale or 1.0

    for (M, n_s, fr, sp), c in zip(tepai_spectra, colors):
        ax.plot(fr, np.abs(sp) / scale, color=c, lw=1.3,
                label=fr"TE-PAI, $N_s={n_s}$, $M_{{TE-PAI}}={M}$")
    ax.plot(ft, np.abs(st) / scale, color="#1f77b4", ls="--", lw=1.3,
            label=fr"Trotter, $N_s={cfg['trotter_shots']}$")
    ax.axvline(gap, color="0.5", ls=":", lw=1,
               label=fr"$\Delta E_{{0,10}} \approx {gap:.2f}$")

    ax.set_xlabel("$E$")
    ax.set_ylabel("$I(E)$, Arbitrary units")
    ax.set_xlim(0, np.pi / cfg["dt"])      # up to the sampling Nyquist frequency
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)
    ax.set_title(f"10-qubit Heisenberg energy spectrum (preset: {preset})")
    fig.tight_layout()

    out = f"paper/fig1_{preset}.png"
    fig.savefig(out, dpi=150)
    print(f"[fig1] saved {out}")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "quick"
    if arg not in PRESETS:
        sys.exit(f"unknown preset {arg!r}; choose from {list(PRESETS)}")
    main(arg)
