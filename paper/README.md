# Paper figures

Scripts that reproduce the figures of *"Low-Resource Quantum Energy Gap
Estimation via Randomization"* from the `pai_shadow` library.

## Fig. 1 — energy spectra of the 10-qubit Heisenberg model

[`fig1_heisenberg_spectrum.py`](fig1_heisenberg_spectrum.py) reproduces the
noise-free comparison of TE-PAI shadow spectroscopy against Trotter-based shadow
spectroscopy. The 10-qubit 1D Heisenberg model is evolved from the superposition
`(|E_0> + |E_10>)/sqrt(2)`, so the dominant spectral peak sits at the transition
gap `dE_{0,10} = E_10 - E_0 ≈ 4.36`. All 3-local Pauli observables are estimated
from classical-shadow snapshots, the top 10% most autocorrelated signals are kept
(Ljung-Box), and a Fourier analysis reveals the peak.

```bash
uv run python paper/fig1_heisenberg_spectrum.py quick   # ~3 min, validates the pipeline
uv run python paper/fig1_heisenberg_spectrum.py paper   # exact paper parameters (heavy: hours)
```

Two presets are defined at the top of the script:

| preset  | N_t | Δ (TE-PAI) | budget / time point | notes |
|---------|-----|-----------|---------------------|-------|
| `quick` | 45  | π/2⁷       | 200 executions      | resolves the gap peak in a few minutes |
| `paper` | 90  | π/2⁷       | 1000 executions     | faithful to the paper; hours on a laptop |

Both run the three TE-PAI sample/shot splits `(M_TE-PAI, N_s)` of a fixed
execution budget plus the Trotter baseline, demonstrating that the recovered
spectrum depends only on the total budget, not on how it is split. The figure is
written to `paper/fig1_<preset>.png` (PNGs are git-ignored).
