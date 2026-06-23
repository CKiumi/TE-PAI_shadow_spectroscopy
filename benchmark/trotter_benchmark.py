"""Benchmark: Trotter time-evolution statevector on qiskit vs qulacs.

Builds a first-order Trotter circuit for the (paper) Heisenberg chain over a
grid of qubit counts and Trotter steps, then times ``statevector`` on each
backend and checks that the two backends agree.

Run:
    uv run python benchmark/trotter_benchmark.py
"""

from __future__ import annotations

import time

import numpy as np

from pai_shadow.backend import get_backend
from pai_shadow.hamil import Heisenberg_Hamil
from pai_shadow.trotter import trotter_circuit

QUBITS = [4, 6, 8, 10]
STEPS = [50, 150, 300]
REPEATS = 3
T = 2.0


def best_time(fn, repeats=REPEATS):
    """Return (best_seconds, result_of_last_call)."""
    best = float("inf")
    out = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return best, out


def fidelity(a, b):
    return abs(np.vdot(a, b)) ** 2


def main():
    qk = get_backend("qiskit")
    ql = get_backend("qulacs")
    print(f"Trotter statevector benchmark  (t={T}, repeats={REPEATS}, best-of)\n")
    header = f"{'nq':>3} {'steps':>6} {'gates':>7} {'depth':>6} " \
             f"{'qiskit[ms]':>11} {'qulacs[ms]':>11} {'speedup':>8} {'fidelity':>9}"
    print(header)
    print("-" * len(header))
    for nq in QUBITS:
        h = Heisenberg_Hamil(nq, 1, 1, 1)
        # fixed initial state with support across the spectrum
        psi0 = np.ones(1 << nq, dtype=complex)
        psi0 /= np.linalg.norm(psi0)
        for steps in STEPS:
            circ = trotter_circuit(h, T, steps, init_state=psi0)
            t_qk, sv_qk = best_time(lambda: qk.statevector(circ))
            t_ql, sv_ql = best_time(lambda: ql.statevector(circ))
            fid = fidelity(sv_qk, sv_ql)
            speed = t_qk / t_ql if t_ql > 0 else float("inf")
            print(f"{nq:>3} {steps:>6} {len(circ):>7} {circ.depth():>6} "
                  f"{t_qk*1e3:>11.1f} {t_ql*1e3:>11.1f} {speed:>7.1f}x {fid:>9.6f}")
    print("\nspeedup = qiskit_time / qulacs_time  (>1 means qulacs faster)")


if __name__ == "__main__":
    main()
