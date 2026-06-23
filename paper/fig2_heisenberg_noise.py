"""Reproduce Fig. 2: effect of depolarizing noise on the energy spectra.

Paper: "Low-Resource Quantum Energy Gap Estimation via Randomization".

Fig. 2 shows that, under gate noise, **TE-PAI shadow spectroscopy** keeps a clear
spectral peak while **Trotter-based shadow spectroscopy** degrades -- because the
TE-PAI circuits are much shallower and so accumulate far less gate noise.

This script reproduces the **reference implementation's** 6-qubit configuration
(the original ``main.py``): a 1D Heisenberg model (Eq. 9, Jx=Jy=Jz=1) evolved from
``|E_0> + |E_40>`` with

    delta = pi/2**6,  Nt = 70,  dt = 2/Nt  (t_max = 2.0),  k = 4 (local Paulis),
    TE-PAI Trotter step size 0.005 capped at N_trotter_max = 150 steps,
    Trotter baseline = 150 steps,  1500 circuit executions per time point.

Depolarizing noise (paper Fig. 2): p1 = 1e-4 (1-qubit), p2 = 1e-3 (2-qubit). The
Heisenberg Hamiltonian has only XX/YY/ZZ terms, so in practice the 2-qubit
channel acts. Four curves are drawn: each method noise-free (dotted) and noisy
(solid).

Run:  uv run python paper/fig2_heisenberg_noise.py [paper|quick]
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

from pai_shadow.hamil import Heisenberg_Hamil
from pai_shadow.circuit import NoiseSpec
from pai_shadow.shadow_spectro import (
    dominant_gap,
    te_pai_shadow_spectroscopy,
    trotter_shadow_spectroscopy,
)

# --- reference implementation's 6-qubit settings (original main.py) ---------- #
N_QUBITS = 6
EXCITED = 40                       # initial state = |E_0> + |E_40>
K_LOCAL = 4
DELTA = np.pi / 2**6
TROTTER_STEP = 0.005              # TE-PAI step size (dt_T), capped at N_TROTTER_MAX
N_TROTTER_MAX = 150
TROTTER_STEPS = 150              # Trotter baseline steps per circuit
SEED = 0
N_JOBS = os.cpu_count()

# depolarizing noise: p1 single-qubit, p2 two-qubit (paper Fig. 2)
NOISE = NoiseSpec(p1=1e-4, p2=1e-3, kind="depolarizing")

PRESETS = {
    # n_t, dt, executions per time point (TE-PAI M x N_s=1; Trotter N_s)
    "paper": dict(n_t=70, dt=2 / 70, M_tepai=1500, n_s=1, trotter_shots=1500),
    "quick": dict(n_t=70, dt=2 / 70, M_tepai=400,  n_s=1, trotter_shots=400),
}


def main(preset: str = "quick") -> None:
    cfg = PRESETS[preset]
    print(f"[fig2] preset={preset!r}  n_jobs={N_JOBS}  params={cfg}")

    H = Heisenberg_Hamil(N_QUBITS, 1.0, 1.0, 1.0)
    e0, e_exc, g, x = H.get_ground_and_excited_state(n=EXCITED)
    init = g + x
    gap = e_exc - e0
    print(f"[fig2] target gap dE_0,{EXCITED} = {gap:.4f}")

    times = np.arange(cfg["n_t"]) * cfg["dt"]
    curves = {}

    def run_trotter(noise, label):
        t0 = time.perf_counter()
        fr, sp = trotter_shadow_spectroscopy(
            H, init, times, n_steps=TROTTER_STEPS, shadow_size=cfg["trotter_shots"],
            k=K_LOCAL, noise=noise, seed=SEED + 1, n_jobs=N_JOBS,
        )
        print(f"[fig2] {label}: peak={dominant_gap(fr, sp):.3f}  ({time.perf_counter()-t0:.1f}s)")
        curves[label] = (fr, sp)

    def run_tepai(noise, label):
        t0 = time.perf_counter()
        fr, sp = te_pai_shadow_spectroscopy(
            H, init, times, delta=DELTA, M=cfg["M_tepai"], n_shots=cfg["n_s"],
            trotter_step=TROTTER_STEP, n_trotter_max=N_TROTTER_MAX,
            k=K_LOCAL, noise=noise, seed=SEED + 2, n_jobs=N_JOBS,
        )
        print(f"[fig2] {label}: peak={dominant_gap(fr, sp):.3f}  ({time.perf_counter()-t0:.1f}s)")
        curves[label] = (fr, sp)

    run_trotter(None, "Trotter, noise-free")
    run_trotter(NOISE, "Trotter, noisy")
    run_tepai(None, "TE-PAI, noise-free")
    run_tepai(NOISE, "TE-PAI, noisy")

    _plot(curves, gap, cfg, preset)


def _plot(curves, gap, cfg, preset) -> None:
    scale = max(np.max(np.abs(sp)) for _, sp in curves.values()) or 1.0
    styles = {
        "Trotter, noise-free": dict(color="#1f77b4", ls=":",  lw=1.3),
        "Trotter, noisy":      dict(color="#1f77b4", ls="-",  lw=1.6),
        "TE-PAI, noise-free":  dict(color="#d62728", ls=":",  lw=1.3),
        "TE-PAI, noisy":       dict(color="#d62728", ls="-",  lw=1.6),
    }
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, (fr, sp) in curves.items():
        ax.plot(fr, np.abs(sp) / scale, label=label, **styles[label])
    ax.axvline(gap, color="0.5", ls="--", lw=1,
               label=fr"$\Delta E_{{0,{EXCITED}}} \approx {gap:.2f}$")
    ax.set_xlabel("$E$")
    ax.set_ylabel("$I(E)$, Arbitrary units")
    ax.set_xlim(0, np.pi / cfg["dt"])
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)
    ax.set_title(f"6-qubit Heisenberg under depolarizing noise (preset: {preset})")
    fig.tight_layout()
    out = f"paper/fig2_{preset}.png"
    fig.savefig(out, dpi=150)
    print(f"[fig2] saved {out}")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "quick"
    if arg not in PRESETS:
        sys.exit(f"unknown preset {arg!r}; choose from {list(PRESETS)}")
    main(arg)
