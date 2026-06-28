"""Fig. 3 generator (self-contained): amplitude-damping comparison at a given shot budget M
and seed. Produces fig3_seed{SEED}_M{M}.npz with the SAME key layout as fig3_seed8.npz /
fig3_seed8_M10000.npz, so regen_fig3_pdf.py can plot it.

Fig. 3 settings: 6-qubit Heisenberg, init |E_0>+|E_10>, Delta=pi/2**6, constant dt_T (N_DIV=500),
amplitude damping p1=0.25e-4/p2=0.25e-3, k=3, n_t=80, t_max=4, DAMPING=0.1, PAD=4.
Seed convention (matches fig3_seed8): base SEED, trotter uses SEED+1, TE-PAI uses SEED+2.

Run:  uv run python paper/fig3/fig3.py <M> [SEED]    (default SEED=80)
"""
from __future__ import annotations
import os, sys, time
import numpy as np
from pai_shadow.hamil import Heisenberg_Hamil
from pai_shadow.circuit import NoiseSpec
from pai_shadow.shadow_spectro import dominant_gap, te_pai_shadow_spectroscopy, trotter_shadow_spectroscopy

HERE = os.path.dirname(os.path.abspath(__file__))
N_QUBITS, EXCITED, K_LOCAL = 6, 10, 3
DELTA, N_DIV, DAMPING, PAD = np.pi / 2**6, 500, 0.1, 4
N_T, T_MAX = 80, 4.0
NOISE = NoiseSpec(p1=0.25e-4, p2=0.25e-3, kind="amplitude_damping")
NJ = os.cpu_count()

M = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 80


def main():
    H = Heisenberg_Hamil(N_QUBITS, 1.0, 1.0, 1.0)
    e0, e_exc, g, x = H.get_ground_and_excited_state(n=EXCITED)
    init = g + x; gap = float(e_exc - e0)
    times = np.linspace(0, T_MAX, N_T); DT = T_MAX / N_DIV; dt = float(times[1] - times[0])
    print(f"[fig3sim] M={M} seed={SEED} gap={gap:.4f} dt_T={DT:.5f}", flush=True)
    curves, D = {}, {}

    def run_tepai(noise, label):
        t0 = time.perf_counter()
        fr, sp, d = te_pai_shadow_spectroscopy(
            H, init, times, delta=DELTA, M=M, n_shots=1, trotter_step=DT, n_trotter_max=None,
            k=K_LOCAL, noise=noise, damping=DAMPING, pad=PAD, seed=SEED + 2, n_jobs=NJ, label=label, return_data=True)
        print(f"[fig3sim] {label}: peak={dominant_gap(fr, sp):.3f} ({time.perf_counter()-t0:.0f}s)", flush=True)
        curves[label] = (fr, sp); D[label] = d

    def run_trotter(noise, label):
        t0 = time.perf_counter()
        fr, sp, d = trotter_shadow_spectroscopy(
            H, init, times, trotter_step=DT, shadow_size=M, k=K_LOCAL, noise=noise,
            density=True, damping=DAMPING, pad=PAD, seed=SEED + 1, n_jobs=NJ, label=label, return_data=True)
        print(f"[fig3sim] {label}: peak={dominant_gap(fr, sp):.3f} ({time.perf_counter()-t0:.0f}s)", flush=True)
        curves[label] = (fr, sp); D[label] = d

    run_tepai(None, "TE-PAI")
    run_tepai(NOISE, "TE-PAI, noisy")
    run_trotter(None, "Trotter")
    run_trotter(NOISE, "Trotter, noisy")

    key = lambda l: l.replace(", ", "_").replace(" ", "_")
    out = os.path.join(HERE, f"fig3_seed{SEED}_M{M}.npz")
    np.savez(out, gap=gap, dt=dt, dt_T=DT, damping=DAMPING, cutoff=4, ljung=True, seed=SEED,
             **{key(l) + "_f": fr for l, (fr, _) in curves.items()},
             **{key(l) + "_s": sp for l, (_, sp) in curves.items()},
             **{key(l) + "_D": D[l] for l in D})
    print(f"[fig3sim] saved {out}", flush=True)


if __name__ == "__main__":
    main()
