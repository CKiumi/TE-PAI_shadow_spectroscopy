"""Reproduce Fig. 2 -- near-equivalent of Hugo Pages' ``main2.py``.

6-qubit Heisenberg, effect of depolarizing noise on the energy spectra: TE-PAI
shadow spectroscopy keeps its peak while the (deeper) Trotter-based spectrum
degrades. Mirrors the reference ``main2.py`` parameters:

    numQs = 6,  init = |E_0> + |E_10>,  1D Heisenberg (Jx=Jy=Jz=1),
    delta = pi/2**5,  k = 3,  Nt = 90,  dt = 0.044,
    TE-PAI: trotter step 0.01 capped at N_trotter_max = 300,  M = 3000,  Ns = 1,
    Trotter: N = 300 steps,  shadow_size = 3000,
    depolarizing noise p1 = 1e-4 (1-qubit), p2 = 1e-3 (2-qubit).

Four curves as in the paper's Fig. 2: TE-PAI (red) and Trotter (blue), each
without noise (solid) and with noise (dashed). Classical-shadow path (= Hugo's
density_matrix=True, statistically equivalent): the noisy snapshots are trajectory
single shots. Time grid follows ``T = linspace(0, Nt*dt, Nt)``.

Run:  uv run python paper/fig2_heisenberg_noise.py [hugo|quick]
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

# --- Hugo main2.py Fig 2 settings ------------------------------------------- #
N_QUBITS = 6
EXCITED = 10                       # initial state = |E_0> + |E_10>
K_LOCAL = 3
DELTA = np.pi / 2**5               # Hugo main2.py value
TROTTER_STEP = 0.01                # TE-PAI step size dt_T
N_TROTTER_MAX = 300                # cap on TE-PAI steps; also the Trotter baseline
SEED = 0
N_JOBS = os.cpu_count()

NOISE = NoiseSpec(p1=1e-4, p2=1e-3, kind="depolarizing")

PRESETS = {
    # n_t, t_max (Hugo: Nt=90, dt=0.044 -> t_max = Nt*dt = 3.96), M, Ns, trotter_shots
    "hugo":  dict(n_t=90, t_max=90 * 0.044, M=3000, n_s=1, trotter_shots=3000),
    "quick": dict(n_t=45, t_max=90 * 0.044, M=3000,  n_s=1, trotter_shots=3000),
}
PRESETS["paper"] = PRESETS["hugo"]   # alias


def main(preset: str = "quick") -> None:
    cfg = PRESETS[preset]
    print(f"[fig2] preset={preset!r}  n_jobs={N_JOBS}  params={cfg}", flush=True)

    H = Heisenberg_Hamil(N_QUBITS, 1.0, 1.0, 1.0)
    e0, e_exc, g, x = H.get_ground_and_excited_state(n=EXCITED)
    init = g + x
    gap = e_exc - e0
    print(f"[fig2] target gap dE_0,{EXCITED} = {gap:.4f}", flush=True)

    times = np.linspace(0, cfg["t_max"], cfg["n_t"])     # Hugo's T grid
    print(f"[fig2] delta=pi/2^5={DELTA:.4f}  trotter_step={TROTTER_STEP}  N_trotter_max={N_TROTTER_MAX}", flush=True)
    curves = {}

    def run_trotter(noise, label, n_steps):
        t0 = time.perf_counter()
        print(f"[fig2] start {label} (N={n_steps}) ...", flush=True)
        fr, sp = trotter_shadow_spectroscopy(
            H, init, times, n_steps=n_steps, shadow_size=cfg["trotter_shots"],
            k=K_LOCAL, noise=noise, density=True, seed=SEED + 1, n_jobs=N_JOBS, label=label,
        )
        print(f"[fig2] {label}: peak={dominant_gap(fr, sp):.3f}  ({time.perf_counter()-t0:.1f}s)", flush=True)
        curves[label] = (fr, sp)

    def run_tepai(noise, label):
        t0 = time.perf_counter()
        print(f"[fig2] start {label} ...", flush=True)
        fr, sp = te_pai_shadow_spectroscopy(
            H, init, times, delta=DELTA, M=cfg["M"], n_shots=cfg["n_s"],
            trotter_step=TROTTER_STEP, n_trotter_max=N_TROTTER_MAX,
            k=K_LOCAL, noise=noise, seed=SEED + 2, n_jobs=N_JOBS, label=label,
        )
        print(f"[fig2] {label}: peak={dominant_gap(fr, sp):.3f}  ({time.perf_counter()-t0:.1f}s)", flush=True)
        curves[label] = (fr, sp)

    run_tepai(None,  "TE-PAI")
    run_tepai(NOISE, "TE-PAI, noisy")
    run_trotter(None,  "Trotter", N_TROTTER_MAX)
    run_trotter(NOISE, "Trotter, noisy", N_TROTTER_MAX)

    dt = float(times[1] - times[0])
    np.savez(f"paper/fig2_{preset}.npz", gap=gap, dt=dt,
             **{lbl.replace(", ", "_").replace(" ", "_") + "_f": fr for lbl, (fr, _) in curves.items()},
             **{lbl.replace(", ", "_").replace(" ", "_") + "_s": sp for lbl, (_, sp) in curves.items()})
    _plot(curves, gap, dt, preset)


def _plot(curves, gap, dt, preset) -> None:
    # Hugo's plot_multiple_data style: red TE-PAI, blue Trotter, green same-depth;
    # solid = noise-free, dashed = noisy. Raw |amplitude| (arbitrary units).
    styles = {
        "TE-PAI":         dict(color="red",  ls="-"),
        "TE-PAI, noisy":  dict(color="red",  ls="--"),
        "Trotter":        dict(color="blue", ls="-"),
        "Trotter, noisy": dict(color="blue", ls="--"),
    }
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, (fr, sp) in curves.items():
        ax.plot(fr, np.abs(sp), label=label, lw=1.5, **styles[label])
    ax.axvline(gap, color="gray", ls="--", lw=1, label=f"Theoretical Energy Gap : {gap:.3f}")
    ax.grid(True, ls="--", color="gray", lw=0.5, alpha=0.7)
    ax.set_xlabel("Frequency (rad/s)", fontsize=14)
    ax.set_ylabel("Amplitude,\n Arbitrary units", fontsize=14)
    ax.set_xlim(0, np.pi / dt)
    ax.set_ylim(bottom=0)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=10, loc="best")
    ax.set_title("TE-PAI vs Trotter shadow spectroscopy with/without noise",
                 fontsize=16, fontweight="bold")
    fig.tight_layout()
    out = f"paper/fig2_{preset}.png"
    fig.savefig(out, dpi=200)
    print(f"[fig2] saved {out}", flush=True)


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "quick"
    if arg not in PRESETS:
        sys.exit(f"unknown preset {arg!r}; choose from {list(PRESETS)}")
    main(arg)
