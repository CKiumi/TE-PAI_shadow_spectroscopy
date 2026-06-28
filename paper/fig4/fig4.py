"""Fig. 4 generator -- TE-PAI vs Trotter on a dominant-ground multi-gap state, with and
without gate noise. Self-contained (restored/merged from the old fig5_trotter_vs_tepai.py
+ fig5_incremental.py); produces paper/fig4/fig4_M4000.npz, which regen_fig4_pdf.py turns
into the paper PDF.

Setup (recipe of arXiv:2212.11036 Fig. 2): dominant-ground initial state
    |psi> = |E_0> + eps * sum_{k>=1} |E_k>,   eps = 0.25,
over the energy levels LEVELS = [0,1,2,4,7,9] of the 5-qubit Heisenberg model. These give
FIVE well-separated dominant excitation gaps  E_k - E_0 = {2.88, 4.48, 6.48, 10.07, 11.71}
(min spacing 1.59, so the gamma budget can resolve them).

Four streams, all at the SAME Delta = pi/2**6 and the SAME fine step dt_T = t_max/N_DIV:
  * TE-PAI            -- drops identity gates, expected gate count ~2|H|_1 t/Delta
                        (independent of dt_T) -> ~2.6k gates.
  * Trotter, fine     -- same fine dt_T, keeps ALL gates -> ~2.5x more gates than TE-PAI.
Under depolarizing gate noise decoherence ~ gate count, so the fine Trotter collapses
first while TE-PAI keeps the fine-dt accuracy at a lower gate budget.

The estimator is a plain shot-mean, so a single M=4000 run with these seeds reproduces the
M=4000 point of the old incremental (batched) accumulation exactly. Seeds are fixed per
stream (100/200/300/400) for reproducibility.

Run:  uv run python paper/fig4/fig4.py
"""
from __future__ import annotations
import os, time
import numpy as np
from pai_shadow.hamil import Heisenberg_Hamil
from pai_shadow.circuit import NoiseSpec
from pai_shadow.shadow_spectro import te_pai_shadow_spectroscopy, trotter_shadow_spectroscopy, dominant_gap

HERE = os.path.dirname(os.path.abspath(__file__))

# --- physics / method settings (from the old fig5_trotter_vs_tepai.py) ---
N_QUBITS, K_LOCAL = 5, 3
LEVELS = [0, 1, 2, 4, 7, 9]        # ground + 5 excited -> 5 dominant gaps
EPS = 0.25                         # sub-dominant weight of the excited levels
DELTA = np.pi / 2**6
N_DIV = 700                        # fine constant dt_T = t_max/N_DIV (theta = 2 dt_T <= Delta)
DAMPING, PAD, CUTOFF = 0.05, 4, 4
NOISE = NoiseSpec(p1=4e-5, p2=4e-4, kind="depolarizing")   # gate noise (depolarizing)

# --- run grid (paper) ---
N_T, T_MAX, M = 120, 6.0, 4000
SEED = {"TE-PAI": 100, "TE-PAI noisy": 200, "Trotter fine": 300, "Trotter fine noisy": 400}
NJ = os.cpu_count()


def main() -> None:
    H = Heisenberg_Hamil(N_QUBITS, 1.0, 1.0, 1.0)
    E, V = H.eigh()
    levels, idx = np.unique(np.round(E, 6), return_index=True)
    sel, sel_E = idx[list(LEVELS)], levels[list(LEVELS)]
    nlev = len(LEVELS)
    w = np.ones(nlev); w[1:] = EPS
    psi = (V[:, sel] * w).sum(axis=1); psi /= np.linalg.norm(psi)
    dom = [round(float(sel_E[k] - sel_E[0]), 4) for k in range(1, nlev)]
    cross = sorted({round(float(sel_E[j] - sel_E[i]), 4)
                    for i in range(1, nlev) for j in range(i + 1, nlev)})

    times = np.linspace(0, T_MAX, N_T)
    DT = T_MAX / N_DIV
    dt = float(times[1] - times[0])
    print(f"[fig4] N={N_QUBITS} levels={LEVELS} dom_gaps={dom} M={M} dt_T={DT:.5f} "
          f"theta={2*DT:.4f}<=Delta={DELTA:.4f}", flush=True)

    curves = {}

    def run_tepai(label, noise):
        t0 = time.perf_counter()
        fr, sp = te_pai_shadow_spectroscopy(
            H, psi, times, delta=DELTA, M=M, n_shots=1, trotter_step=DT, n_trotter_max=None,
            k=K_LOCAL, noise=noise, ljung=True, cutoff=CUTOFF, damping=DAMPING, pad=PAD,
            seed=SEED[label], n_jobs=NJ, label=label)
        curves[label] = (fr, sp)
        print(f"[fig4] {label}: peak={dominant_gap(fr, sp):.3f} ({time.perf_counter()-t0:.0f}s)", flush=True)

    def run_trotter(label, noise):
        t0 = time.perf_counter()
        fr, sp = trotter_shadow_spectroscopy(
            H, psi, times, trotter_step=DT, shadow_size=M, k=K_LOCAL, noise=noise,
            density=(noise is not None), ljung=True, cutoff=CUTOFF, damping=DAMPING, pad=PAD,
            seed=SEED[label], n_jobs=NJ, label=label)
        curves[label] = (fr, sp)
        print(f"[fig4] {label}: peak={dominant_gap(fr, sp):.3f} ({time.perf_counter()-t0:.0f}s)", flush=True)

    run_tepai("TE-PAI", None)
    run_tepai("TE-PAI noisy", NOISE)
    run_trotter("Trotter fine", None)
    run_trotter("Trotter fine noisy", NOISE)

    key = lambda l: l.replace(" ", "_")
    out = os.path.join(HERE, "fig4_M4000.npz")
    np.savez(out, dt=dt, M=M, dom=np.array(dom), cross=np.array(cross),
             **{key(l) + "_f": fr for l, (fr, _) in curves.items()},
             **{key(l) + "_s": sp for l, (_, sp) in curves.items()})
    print(f"[fig4] saved {out}", flush=True)


if __name__ == "__main__":
    main()
