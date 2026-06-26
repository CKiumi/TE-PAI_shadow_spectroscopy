# Figure 5 — TE-PAI vs Trotter shadow spectroscopy on a multi-gap state (under noise)

Paper figure: **`fig5.pdf`** — two panels, noiseless (left) and with gate noise (right),
TE-PAI (red) vs Trotter (blue), each curve normalised to its own peak (`I(E)/max`).

Regenerate from the cached data with:

```bash
uv run python paper/fig5/regen_fig5_pdf.py        # -> paper/fig5/fig5.pdf
```

(reads `inc/fig5_incremental_M4000.npz`).

## Settings

| | |
|---|---|
| **System** | 5-qubit Heisenberg chain, `Jx = Jy = Jz = 1`, `‖H‖₁ = 12` |
| **Initial state** | dominant-ground `|ψ₀⟩ = |E₀⟩ + ε·Σ_{k≥1}|E_k⟩`, `ε = 0.25`, levels `[0,1,2,4,7,9]` |
| **Dominant gaps** | `E_k − E₀ = {2.88, 4.48, 6.48, 10.07, 11.71}` |
| **Observables** | 3-local Pauli classical shadows (`k = 3`) |
| **TE-PAI** | `Δ = π/2⁶ = 0.0491`, `M = 4000`, `n_s = 1` |
| **Trotter** | same step width `δt` (same accuracy), exact-noisy density matrix for the noisy curve, `shots = 4000` |
| **Step (both)** | constant `δt = 0.00857` (`N_DIV = 700`), `θ = 2δt = 0.0171 ≤ Δ` |
| **Time grid** | `t ∈ [0, 6]`, `N_t = 120`, `dt = 0.0504` |
| **Gate noise** | depolarizing, `p₁ = 4×10⁻⁵`, `p₂ = 4×10⁻⁴` |
| **Post-processing** | standardise → Ljung-Box → taper `e^(−0.05 t)` → cross-correlation → SVD (`cutoff = 4`) → FFT (zero-pad ×4) |
| **Normalisation** | per-curve (each curve / its own peak) |
| **RNG / accumulation** | computed incrementally in shot batches; per-entry data matrix `D` is a plain shot-mean, so cumulative `D = Σ Mᵢ Dᵢ / Σ Mᵢ` is exact. D matrices are saved for M = 1000/2000/3000 (`inc/fig5_incremental_M{1000,2000,3000}.npz`); M = 4000–10000 store spectra only. |

### Gate-count / overhead comparison (at `t_max = 6`)

| quantity | TE-PAI | Trotter |
|---|---|---|
| Trotter steps (`δt = 0.0086`) | 700 | 700 |
| circuit gates | ≈ 2.9×10³ (drops identities, `E[ν] = csc Δ (3−cos Δ)‖H‖₁ t`) | ≈ 8.4×10³ |
| squared sampling overhead `Γ²` | `exp[2 t ‖H‖₁ tan(Δ/2)] ≈ 34` | 1 |

## Result / claim

Both methods use the same `δt` (same Trotter accuracy); TE-PAI is ~3× shallower. The
**robust, seed-independent claim** is: under depolarizing gate noise the TE-PAI spectrum
**retains the gap structure** — its intensity stays above Trotter's at every dominant gap
(red above blue, right panel) — whereas Trotter loses it (collapses toward a low-frequency
decoherence peak).

Do **not** claim "the dominant peak sits at the gap": at N = 5, p₂ = 4×10⁻⁴ that is
seed-dependent (TE-PAI's dominant peak is at a gap in ~2/5 seeds; it always retains more
gap structure than Trotter, which is the part that is robust). Lowering the noise makes
the dominant-peak statement hold across seeds.

## Caveat — the high-frequency gaps (10.07, 11.71)

These two gaps are **real** (they have substantial spectral weight; a naive `Σ_i|FFT(S_i)|²`
power spectrum shows them at full strength), but they lie in **sub-dominant SVD modes that the
`cutoff = 4` truncation discards** — so the clean (exact / Trotter) spectra leave them at the
floor. TE-PAI's structured sampling noise re-surfaces them (independent white noise injected
into the exact data does **not** reproduce this — verified in `verify/`). Their heights are
therefore not a faithful intensity. Recommendation: **do not present 10.07/11.71 as resolved
gaps** in the paper; restrict claims to the three cleanly-resolved gaps (2.88, 4.48, 6.48), or
raise the cutoff for all methods if the high gaps are wanted. See `verify/` for the full analysis
(`verify_highfreq`, `verify_cutoff`, `verify_noise_injection`, `verify_whitenoise_reproduce`).
