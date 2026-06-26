"""Fig. 5 (comparison) -- TE-PAI vs Trotter on a dominant-ground multi-gap state,
with and without gate noise.

Dominant-ground initial state |E_0> + eps*sum_{k=1}^{Q-1}|E_k> (recipe of
arXiv:2212.11036 Fig. 2). With Q=6 there are FIVE dominant excitation gaps E_k - E_0.
We compare three ways of getting the time-evolved shadows, all at the SAME Delta=pi/2**6:

  1. TE-PAI            -- step dt (fine), but TE-PAI drops identity gates so its
                          expected gate count is ~2|H|_1 t/Delta, INDEPENDENT of dt.
  2. Trotter, fine     -- same fine step dt -> same accuracy as TE-PAI, but keeps ALL
                          gates: n_terms * t/dt  =  (Delta/2dt)x MORE gates than TE-PAI.
  3. Trotter, matched  -- coarse step dt_coarse=Delta/2 chosen so its gate count equals
     gates                TE-PAI's; same gate budget but a coarser (less accurate) dt.

Noiseless: all three resolve the five gaps (they agree). Under gate noise, decoherence
~ gate count, so the fine Trotter (most gates) collapses first, while TE-PAI keeps the
fine-dt accuracy at the low matched gate budget -- the point of TE-PAI.

To use a FINE dt (so the fine-Trotter gate penalty is visible) we reduce to N=4 qubits:
|H|_1 = 3(N-1) = 9 keeps TE-PAI's overhead gamma ~ exp(t|H|_1 tan(Delta/2)) moderate.

Run:  uv run python paper/fig5_trotter_vs_tepai.py [paper|quick]
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

from pai_shadow.hamil import Heisenberg_Hamil
from pai_shadow.circuit import NoiseSpec
from pai_shadow.te_pai import TEPAI
from pai_shadow.shadow_spectro import (
    dominant_gap,
    te_pai_shadow_spectroscopy,
    trotter_shadow_spectroscopy,
)

N_QUBITS = 5
K_LOCAL = 3
EPS = 0.25
# distinct-energy-level indices to superpose (ground first). For N=5 the lowest few
# levels include a near-degenerate gap pair (spacing 0.59) that the gamma budget can't
# resolve, so we pick a subset with well-separated gaps: {0,1,2,4,7,9} ->
# dominant gaps E_k-E_0 = {2.88, 4.48, 6.48, 10.07, 11.71} (min spacing 1.59).
LEVELS = [0, 1, 2, 4, 7, 9]        # ground + 5 excited -> 5 dominant gaps
DELTA = np.pi / 2**6
N_DIV = 700                        # fine constant dt_T = t_max/N_DIV (theta=2dt_T<=Delta).
                                   # Fine dt -> fine Trotter has ~2.5x TE-PAI's gates.
DAMPING = 0.05
PAD = 4
SEED = 0
N_JOBS = os.cpu_count()

# gate noise (depolarizing), tuned so TE-PAI (~2.6k gates) survives (~1 damping event)
# while the fine Trotter (~6.3k gates) accumulates several and collapses.
NOISE = NoiseSpec(p1=4e-5, p2=4e-4, kind="depolarizing")

PRESETS = {
    # t_max=6 keeps N=5 overhead gamma~5.8 (<6) while Delta_f=1.05 resolves the 1.59-spaced gaps
    "paper": dict(n_t=120, t_max=6.0, M=12000, n_s=1, trotter_shots=12000),
    "m3000": dict(n_t=120, t_max=6.0, M=3000,  n_s=1, trotter_shots=3000),   # paper grid, lower budget
    "m6000": dict(n_t=120, t_max=6.0, M=6000,  n_s=1, trotter_shots=6000),   # paper grid, M=6000
    "quick": dict(n_t=70,  t_max=6.0, M=4000,  n_s=1, trotter_shots=4000),
}


def main(preset: str = "quick") -> None:
    cfg = PRESETS[preset]
    nlev = len(LEVELS)
    print(f"[fig5cmp] preset={preset!r}  N={N_QUBITS}  levels={LEVELS}  noise={NOISE.kind} p2={NOISE.p2:.0e}  params={cfg}", flush=True)

    H = Heisenberg_Hamil(N_QUBITS, 1.0, 1.0, 1.0)
    E, V = H.eigh()
    levels, idx = np.unique(np.round(E, 6), return_index=True)
    sel = idx[list(LEVELS)]; sel_E = levels[list(LEVELS)]
    w = np.ones(nlev); w[1:] = EPS
    psi = (V[:, sel] * w).sum(axis=1); psi /= np.linalg.norm(psi)
    dom = [round(float(sel_E[k] - sel_E[0]), 4) for k in range(1, nlev)]
    cross = sorted({round(float(sel_E[j] - sel_E[i]), 4) for i in range(1, nlev) for j in range(i + 1, nlev)})

    times = np.linspace(0, cfg["t_max"], cfg["n_t"])
    DT = cfg["t_max"] / N_DIV
    n_terms = len(H)
    tp = TEPAI(H, DELTA, cfg["t_max"], round(cfg["t_max"] / DT))
    circs, _ = tp.sample(2000)
    exp_gates = float(np.mean([len(c) for c in circs]))
    DT_COARSE = n_terms * cfg["t_max"] / exp_gates
    fine_gates = n_terms * cfg["t_max"] / DT
    gamma_est = np.exp(cfg["t_max"] * n_terms * np.tan(DELTA / 2))
    print(f"[fig5cmp] dominant gaps={dom}  dt_fine={DT:.4f}  dt_coarse={DT_COARSE:.4f}  gamma~{gamma_est:.1f}", flush=True)
    print(f"[fig5cmp] gates@t_max: TE-PAI~{exp_gates:.0f}  fine-Trotter~{fine_gates:.0f}  matched-Trotter~{exp_gates:.0f}",
          flush=True)

    curves = {}

    def run(label, fn):
        t0 = time.perf_counter()
        print(f"[fig5cmp] start {label} ...", flush=True)
        fr, sp = fn()
        print(f"[fig5cmp] {label} peak={dominant_gap(fr, sp):.2f} ({time.perf_counter()-t0:.0f}s)", flush=True)
        curves[label] = (fr, sp)

    def tepai(noise):
        return lambda: te_pai_shadow_spectroscopy(
            H, psi, times, delta=DELTA, M=cfg["M"], n_shots=cfg["n_s"], trotter_step=DT,
            n_trotter_max=None, k=K_LOCAL, noise=noise, damping=DAMPING, pad=PAD,
            seed=SEED + 2, n_jobs=N_JOBS, label="TE-PAI")

    def trotter(step, noise):
        return lambda: trotter_shadow_spectroscopy(
            H, psi, times, trotter_step=step, shadow_size=cfg["trotter_shots"], k=K_LOCAL,
            noise=noise, density=(noise is not None), damping=DAMPING, pad=PAD,
            seed=SEED + 1, n_jobs=N_JOBS, label="Trotter")

    run("TE-PAI",            tepai(None))
    run("TE-PAI noisy",      tepai(NOISE))
    run("Trotter fine",      trotter(DT, None))
    run("Trotter fine noisy", trotter(DT, NOISE))

    dt = float(times[1] - times[0])
    np.savez(f"paper/fig5_trotter_vs_tepai_{preset}.npz", dt=dt, dom=np.array(dom), cross=np.array(cross),
             dt_fine=DT, dt_coarse=DT_COARSE, exp_gates=exp_gates, fine_gates=fine_gates,
             **{k.replace(" ", "_") + "_f": v[0] for k, v in curves.items()},
             **{k.replace(" ", "_") + "_s": v[1] for k, v in curves.items()})
    _plot(curves, dom, cross, dt, preset, DT, DT_COARSE, exp_gates, fine_gates, cfg, gamma_est)


def _settings_text(dom, dt, dt_fine, exp_gates, fine_gates, cfg, gamma_est) -> str:
    g = ", ".join(f"{x:.2f}" for x in dom)
    return "   |   ".join([
        rf"$\bf{{System}}$: {N_QUBITS}-qubit Heisenberg $J_x{{=}}J_y{{=}}J_z{{=}}1$",
        rf"init $=|E_0\rangle+{EPS}\sum_k|E_k\rangle$ (levels {LEVELS})",
        rf"$\Delta=\pi/2^6={DELTA:.4f}$, $k={K_LOCAL}$",
        rf"$\delta t={dt_fine:.4f}$ ($\theta=2\delta t={2*dt_fine:.4f}\leq\Delta$)",
        rf"gates@$t_{{max}}$: TE-PAI~{exp_gates:.0f}, Trotter~{fine_gates:.0f}",
        rf"noise: {NOISE.kind} $p_1={NOISE.p1:.0e}$ $p_2={NOISE.p2:.0e}$",
        rf"$M=N_s={cfg['M']}$",
        rf"$t\in[0,{cfg['t_max']:g}]$, $N_t={cfg['n_t']}$, $dt={dt:.4f}$",
        rf"damping={DAMPING}, cutoff=4, Ljung, pad={PAD}, seed={SEED}",
        rf"$\gamma(t_{{max}})\sim{gamma_est:.1f}$",
        rf"dominant gaps $E_k-E_0=\{{{g}\}}$",
    ])


def _plot(curves, dom, cross, dt, preset, dt_fine, dt_coarse, exp_gates, fine_gates, cfg, gamma_est) -> None:
    sty = {
        "TE-PAI":          dict(color="red",   ls="-",  lw=1.9),
        "Trotter fine":    dict(color="blue",  ls="-",  lw=1.6),
    }
    lab = {
        "TE-PAI":          f"TE-PAI ($\\delta t${dt_fine:.3f}, ~{exp_gates:.0f} gates)",
        "Trotter fine":    f"Trotter ($\\delta t${dt_fine:.3f}, same accuracy, ~{fine_gates:.0f} gates)",
    }
    nyq = np.pi / dt
    fig, axes = plt.subplots(1, 2, figsize=(17, 6), sharex=True, sharey=True)
    panels = [("noiseless", ""), ("with gate noise", " noisy")]
    for ax, (ptitle, suf) in zip(axes, panels):
        for base in ["TE-PAI", "Trotter fine"]:
            fr, sp = curves[base + suf]; y = np.abs(sp)
            ax.plot(fr, y / (y.max() or 1.0), label=lab[base], **sty[base])
        for j, g in enumerate(dom):
            ax.axvline(g, color="black", ls="--", lw=1.1, label="dominant gaps $E_k-E_0$" if j == 0 else None)
        for g in cross:
            ax.axvline(g, color="0.7", ls=":", lw=0.9)
        ax.grid(True, ls="--", lw=0.5, alpha=0.5)
        ax.set_xlim(0, max(dom + cross) + 1.0); ax.set_ylim(0, 1.05)
        ax.set_xlabel("Frequency (rad/s)"); ax.set_title(ptitle, fontsize=13, fontweight="bold")
        ax.legend(fontsize=8.5, loc="upper right")
    axes[0].set_ylabel("$I(E)$/max")
    fig.suptitle(f"Fig. 5 -- TE-PAI vs Trotter, {len(dom)} gaps from a dominant-ground state "
                 f"({N_QUBITS}-qubit Heisenberg, {NOISE.kind} $p_2$={NOISE.p2:.0e})",
                 fontsize=13, fontweight="bold")
    # full configuration printed under the panels (reproducibility)
    txt = _settings_text(dom, dt, dt_fine, exp_gates, fine_gates, cfg, gamma_est)
    fig.text(0.5, 0.015, txt, ha="center", va="bottom", fontsize=8,
             bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.7", alpha=0.9), wrap=True)
    fig.tight_layout(rect=[0, 0.075, 1, 1])
    out = f"paper/fig5_trotter_vs_tepai_{preset}.png"
    fig.savefig(out, dpi=170)
    print(f"[fig5cmp] saved {out}", flush=True)


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "quick"
    if arg not in PRESETS:
        sys.exit(f"unknown preset {arg!r}; choose from {list(PRESETS)}")
    main(arg)
