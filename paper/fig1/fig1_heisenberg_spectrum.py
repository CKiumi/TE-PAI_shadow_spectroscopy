"""Reproduce Fig. 1 -- near-equivalent of Hugo Pages' ``main.py``.

10-qubit Heisenberg energy spectrum from TE-PAI shadow spectroscopy (three
sample/snapshot splits of a fixed 1000-execution budget) versus Trotter-based
shadow spectroscopy. Mirrors the reference ``main.py`` parameters:

    numQs = 10,  delta = pi/2**7,  k = 3,  Nt = 90,  dt = 10/90,
    TE-PAI: trotter step 0.001 capped at N_trotter_max = 650,
            (M_sample, Ns) in {(1000,1), (500,2), (250,4)}  -> 1000 executions,
    Trotter: N_Trotter_steps = 500, shadow_size = 1000,
    init state = |E_0> + |E_10>  (ground + 10th excited).

Classical-shadow path (= Hugo's density_matrix=False): one random measurement per
circuit-execution, weighted by the TE-PAI quasiprobability. Time grid follows
Hugo's ``T = linspace(0, Nt*dt, Nt)``.

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

# --- Hugo main.py Fig 1 settings -------------------------------------------- #
N_QUBITS = 10
EXCITED = 10                       # initial state = |E_0> + |E_10>
DELTA = np.pi / 2**7
K_LOCAL = 3
TROTTER_STEP = 0.001               # TE-PAI step size dt_T
N_TROTTER_MAX = 650                # cap on TE-PAI steps
TROTTER_STEPS = 500                # Trotter baseline steps
TROTTER_SHOTS = 1000               # Trotter shadow snapshots
SEED = 0
N_JOBS = os.cpu_count()

# fixed 1000-execution budget split three ways: M_sample * Ns = 1000
TE_PAI_CONFIGS = [
    dict(M=1000, n_s=1),
    dict(M=500,  n_s=2),
    dict(M=250,  n_s=4),
]

PRESETS = {
    # n_t, t_max  (Hugo: Nt=90, dt=10/90 -> t_max = Nt*dt = 10)
    "paper": dict(n_t=90, t_max=10.0),
    "quick": dict(n_t=45, t_max=10.0),
}


def main(preset: str = "quick") -> None:
    cfg = PRESETS[preset]
    print(f"[fig1] preset={preset!r}  n_jobs={N_JOBS}", flush=True)

    H = Heisenberg_Hamil(N_QUBITS, 1.0, 1.0, 1.0)
    e0, e_exc, g, x = H.get_ground_and_excited_state(n=EXCITED)
    init = g + x
    gap = e_exc - e0
    print(f"[fig1] E_0={e0:.4f}  E_{EXCITED}={e_exc:.4f}  gap dE_0,{EXCITED}={gap:.4f}", flush=True)

    times = np.linspace(0, cfg["t_max"], cfg["n_t"])     # Hugo's T grid
    curves = []  # (label, freq, spectrum)

    for c in TE_PAI_CONFIGS:
        label = f"TE-PAI, M={c['M']}, Ns={c['n_s']}"
        t0 = time.perf_counter()
        print(f"[fig1] start {label} ...", flush=True)
        fr, sp = te_pai_shadow_spectroscopy(
            H, init, times, delta=DELTA, M=c["M"], n_shots=c["n_s"],
            trotter_step=TROTTER_STEP, n_trotter_max=N_TROTTER_MAX,
            k=K_LOCAL, noise=None, seed=SEED + 1, n_jobs=N_JOBS, label=label,
        )
        print(f"[fig1] {label}: peak={dominant_gap(fr, sp):.3f}  ({time.perf_counter()-t0:.1f}s)", flush=True)
        curves.append((label, fr, sp))

    label = f"Trotter, Ns={TROTTER_SHOTS}"
    t0 = time.perf_counter()
    print(f"[fig1] start {label} ...", flush=True)
    fr, sp = trotter_shadow_spectroscopy(
        H, init, times, n_steps=TROTTER_STEPS, shadow_size=TROTTER_SHOTS,
        k=K_LOCAL, noise=None, seed=SEED + 2, n_jobs=N_JOBS, label=label,
    )
    print(f"[fig1] {label}: peak={dominant_gap(fr, sp):.3f}  ({time.perf_counter()-t0:.1f}s)", flush=True)
    curves.append((label, fr, sp))

    dt = float(times[1] - times[0])
    np.savez(f"paper/fig1_{preset}.npz", gap=gap, dt=dt,
             labels=[c[0] for c in curves],
             **{f"f{i}": c[1] for i, c in enumerate(curves)},
             **{f"s{i}": c[2] for i, c in enumerate(curves)})
    _plot(curves, gap, dt, preset)


def _plot(curves, gap, dt, preset) -> None:
    colors = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (label, fr, sp) in enumerate(curves):
        ls = ":" if label.startswith("Trotter") else "-"
        ax.plot(fr, np.abs(sp), label=label, color=colors[i % len(colors)], ls=ls, lw=1.5)
    ax.axvline(gap, color="0.5", ls="--", lw=1, label=fr"$\Delta E_{{0,{EXCITED}}} \approx {gap:.2f}$")
    ax.set_xlabel("$E$")
    ax.set_ylabel("$I(E)$, Arbitrary units")
    ax.set_xlim(0, np.pi / dt)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)
    ax.set_title(f"10-qubit Heisenberg spectrum: TE-PAI vs Trotter (preset: {preset})")
    fig.tight_layout()
    out = f"paper/fig1_{preset}.png"
    fig.savefig(out, dpi=200)
    print(f"[fig1] saved {out}", flush=True)


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "quick"
    if arg not in PRESETS:
        sys.exit(f"unknown preset {arg!r}; choose from {list(PRESETS)}")
    main(arg)
