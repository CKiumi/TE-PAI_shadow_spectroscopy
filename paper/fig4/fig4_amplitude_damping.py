"""Fig. 4 -- amplitude-damping (T1) version of the Fig. 3 constant-dt comparison.

Identical setup to Fig. 3 (constant step width for both TE-PAI and Trotter,
Delta = pi/2**6, equal measurement budget), but the gate noise is **amplitude
damping** (T1 relaxation) instead of depolarizing. Amplitude damping is
**non-unital**: errors all push the state toward |0...0>, so the effect on the
signal is qualitatively different from (and at the same per-gate rate, stronger
than) the unital depolarizing channel. Showing the TE-PAI robustness survives
this channel demonstrates it is not specific to depolarizing noise.

    numQs = 6,  init = |E_0> + |E_10>,  Heisenberg (Jx=Jy=Jz=1),
    delta = pi/2**6,  k = 3,  Nt = 90,  t_max = 3.96,
    constant step width dt_T = t_max / N_DIV (same for TE-PAI and Trotter),
    M = Ns_Trotter = 10000,  amplitude damping p1 = 1e-4 / p2 = 1e-3,
    Trotter evaluated via the exact noisy density matrix (density=True).

Four curves: TE-PAI (red) and Trotter (blue), each noise-free (solid) / noisy
(dashed), plotted raw / unnormalised (Hugo's convention, shared vertical axis).

Run:  uv run python paper/fig4_amplitude_damping.py [paper|quick]
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

# --- parameters (same as Fig. 3 except the noise channel) ------------------- #
N_QUBITS = 6
EXCITED = 10                       # initial state = |E_0> + |E_10>
K_LOCAL = 3
DELTA = np.pi / 2**6               # clean-shadow regime (gamma stays ~3-8)
N_DIV = 500                        # divisions of [0, t_max] -> constant dt_T = t_max / N_DIV
DAMPING = 0.1                      # post-processing time taper exp(-DAMPING * t), Hugo's value.
                                   # With the shorter t_max=3.5 the TE-PAI gap peak already
                                   # dominates its (reduced) low-frequency decay mode, so no heavy
                                   # taper is needed -- keep the standard 0.1 for consistency.
PAD = 4                            # FFT zero-padding factor: interpolates onto a finer frequency
                                   # grid for a smooth curve (no extra real resolution). 1=raw
                                   # (coarse), 2-4=balanced, 8=very smooth -- tune to taste.
SEED = 0
N_JOBS = os.cpu_count()

# amplitude damping (T1): non-unital, biases the state toward |0...0>
NOISE = NoiseSpec(p1=0.25*1e-4, p2=0.25*1e-3, kind="amplitude_damping")

PRESETS = {
    # n_t, t_max, M, Ns, trotter_shots.  Shorter t_max = 3.5 (vs Hugo's 3.96): amplitude
    # damping accumulates a strong low-frequency decay mode, and stopping the evolution
    # before the signal fully decoheres keeps the TE-PAI gap peak as the dominant feature
    # (the deep Trotter has already collapsed). dt ~ 0.0443 (same grid spacing as Hugo).
    "paper": dict(n_t=80, t_max=4, M=3000, n_s=1, trotter_shots=3000),
    "quick": dict(n_t=40, t_max=2.5, M=3000,  n_s=1, trotter_shots=3000),
}


def main(preset: str = "quick") -> None:
    cfg = PRESETS[preset]
    print(f"[fig4] preset={preset!r}  n_jobs={N_JOBS}  params={cfg}", flush=True)

    H = Heisenberg_Hamil(N_QUBITS, 1.0, 1.0, 1.0)
    e0, e_exc, g, x = H.get_ground_and_excited_state(n=EXCITED)
    init = g + x
    gap = e_exc - e0
    print(f"[fig4] target gap dE_0,{EXCITED} = {gap:.4f}", flush=True)

    times = np.linspace(0, cfg["t_max"], cfg["n_t"])
    DT = cfg["t_max"] / N_DIV                      # constant step width, both methods
    theta = 2 * DT                                 # TE-PAI angle (coef=1); must be <= DELTA
    print(f"[fig4] amplitude_damping  delta=pi/2^6={DELTA:.4f}  constant dt_T={DT:.5f}  "
          f"theta=2*dt_T={theta:.5f} (<= delta: {theta <= DELTA})", flush=True)
    curves = {}
    data_mats = {}                                 # label -> raw (Nt, No) data matrix D

    def run_trotter(noise, label):
        t0 = time.perf_counter()
        print(f"[fig4] start {label} ...", flush=True)
        fr, sp, D = trotter_shadow_spectroscopy(
            H, init, times, trotter_step=DT, shadow_size=cfg["trotter_shots"],
            k=K_LOCAL, noise=noise, density=True, damping=DAMPING, pad=PAD,
            seed=SEED + 1, n_jobs=N_JOBS, label=label, return_data=True,
        )
        print(f"[fig4] {label}: peak={dominant_gap(fr, sp):.3f}  ({time.perf_counter()-t0:.1f}s)", flush=True)
        curves[label] = (fr, sp); data_mats[label] = D

    def run_tepai(noise, label):
        t0 = time.perf_counter()
        print(f"[fig4] start {label} ...", flush=True)
        fr, sp, D = te_pai_shadow_spectroscopy(
            H, init, times, delta=DELTA, M=cfg["M"], n_shots=cfg["n_s"],
            trotter_step=DT, n_trotter_max=None,   # no cap -> true constant dt_T
            k=K_LOCAL, noise=noise, damping=DAMPING, pad=PAD,
            seed=SEED + 2, n_jobs=N_JOBS, label=label, return_data=True,
        )
        print(f"[fig4] {label}: peak={dominant_gap(fr, sp):.3f}  ({time.perf_counter()-t0:.1f}s)", flush=True)
        curves[label] = (fr, sp); data_mats[label] = D

    run_tepai(None,  "TE-PAI")
    run_tepai(NOISE, "TE-PAI, noisy")
    run_trotter(None,  "Trotter")
    run_trotter(NOISE, "Trotter, noisy")

    dt = float(times[1] - times[0])
    key = lambda lbl: lbl.replace(", ", "_").replace(" ", "_")
    # Save the post-processed spectra AND the raw pre-post-processing data matrices D
    # (one per curve) plus the post-processing settings, so different damping/cutoff
    # can be compared later from the cache without re-simulating (see compare_damping.py).
    np.savez(f"paper/fig4_{preset}.npz", gap=gap, dt=dt, dt_T=DT,
             damping=DAMPING, cutoff=4, ljung=True,
             **{key(lbl) + "_f": fr for lbl, (fr, _) in curves.items()},
             **{key(lbl) + "_s": sp for lbl, (_, sp) in curves.items()},
             **{key(lbl) + "_D": data_mats[lbl] for lbl in data_mats})
    print(f"[fig4] cached raw data matrices D in paper/fig4_{preset}.npz", flush=True)
    _plot(curves, gap, dt, preset, cfg)


def _settings_text(gap, dt, cfg) -> str:
    """Full configuration string embedded in the figure (every setting, for reproducibility)."""
    dt_T = cfg["t_max"] / N_DIV
    return "\n".join([
        r"$\bf{System}$: 6 qubits, Heisenberg $J_x{=}J_y{=}J_z{=}1$, init $=|E_0\rangle+|E_{10}\rangle$",
        rf"$\bf{{Target}}$: $\Delta E_{{0,10}}={gap:.4f}$",
        rf"$\bf{{TE\text{{-}}PAI}}$: $\Delta=\pi/2^6={DELTA:.4f}$, $k={K_LOCAL}$, "
        rf"$M={cfg['M']}\times n_s={cfg['n_s']}$",
        rf"$\bf{{Trotter}}$: exact noisy density matrix, shots$={cfg['trotter_shots']}$",
        rf"$\bf{{Step}}$: constant $dt_T={dt_T:.5f}$ ($N_{{div}}={N_DIV}$), "
        rf"$\theta=2dt_T={2*dt_T:.4f}\leq\Delta$",
        rf"$\bf{{Time\ grid}}$: $t\in[0,{cfg['t_max']:g}]$, $N_t={cfg['n_t']}$, $dt={dt:.4f}$",
        rf"$\bf{{Noise}}$: {NOISE.kind} ($T_1$), $p_1={NOISE.p1:.2e}$, $p_2={NOISE.p2:.2e}$",
        rf"$\bf{{Post}}$: std$\to$Ljung-Box$\to$taper $e^{{-{DAMPING}t}}\to$corr$\to$SVD, "
        rf"FFT pad$={PAD}$, seed$={SEED}$",
    ])


def _plot(curves, gap, dt, preset, cfg) -> None:
    styles = {
        "TE-PAI":         dict(color="red",  ls="-"),
        "TE-PAI, noisy":  dict(color="red",  ls="--"),
        "Trotter":        dict(color="blue", ls="-"),
        "Trotter, noisy": dict(color="blue", ls="--"),
    }
    # raw / no normalisation (Hugo's plotting convention): all four curves share the
    # same vertical axis in arbitrary units. The classical-shadow standardisation already
    # puts the per-observable scales on a comparable footing, so for this setup the raw
    # amplitudes land close (all peaks ~1). The collapse shows directly: TE-PAI noisy keeps
    # its gap peak, while Trotter noisy's dominant feature is the low-frequency decoherence
    # mode (and is the tallest of all). (Cross-method raw heights carry an estimator-variance
    # artifact in general -- see fig4_norm_compare.py -- but here they are consistent.)
    fig, ax = plt.subplots(figsize=(10, 6))
    for label in ["TE-PAI", "TE-PAI, noisy", "Trotter", "Trotter, noisy"]:
        fr, sp = curves[label]
        ax.plot(fr, np.abs(sp), label=label, lw=1.5, **styles[label])
    ax.axvline(gap, color="gray", ls="--", lw=1, label=f"Theoretical Energy Gap : {gap:.3f}")
    ax.grid(True, ls="--", color="gray", lw=0.5, alpha=0.7)
    ax.set_xlabel("Frequency (rad/s)", fontsize=14)
    ax.set_ylabel("$I(E)$  (raw, arbitrary units)", fontsize=13)
    ax.set_xlim(0, np.pi / dt)
    ax.set_ylim(bottom=0)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=10, loc="upper right")
    ax.set_title("TE-PAI vs Trotter shadow spectroscopy (constant dt, amplitude damping)",
                 fontsize=16, fontweight="bold")
    # embed the full configuration in the figure (reproducibility)
    ax.text(0.975, 0.62, _settings_text(gap, dt, cfg), transform=ax.transAxes,
            fontsize=8.5, va="top", ha="right",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.7", alpha=0.9))
    fig.tight_layout()
    out = f"paper/fig4_{preset}.png"
    fig.savefig(out, dpi=200)
    print(f"[fig4] saved {out}", flush=True)


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "quick"
    if arg not in PRESETS:
        sys.exit(f"unknown preset {arg!r}; choose from {list(PRESETS)}")
    main(arg)
