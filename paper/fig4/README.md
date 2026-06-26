# Figure 4 — TE-PAI vs Trotter shadow spectroscopy under amplitude-damping noise

Paper figures: **`fig4-1.pdf`** (M = 3000) and **`fig4-2.pdf`** (M = 10000), each a raw
(unnormalised, Hugo convention) plot of the spectral intensity `I(E)` with four curves:
TE-PAI (red) and Trotter (blue), each noise-free (solid) / noisy (dashed).

Regenerate from the cached data with:

```bash
uv run python paper/fig4/regen_fig4_pdf.py     # -> fig4-1.{pdf,png}, fig4-2.{pdf,png}
```

(reads `fig4_seed8.npz` / `fig4_seed8_M10000.npz`, which also store the raw data matrices `D`).

## Settings (moved here from the in-figure settings box)

| | |
|---|---|
| **System** | 6 qubits, Heisenberg chain, `Jx = Jy = Jz = 1`, `‖H‖₁ = 15` |
| **Initial state** | `|ψ₀⟩ = |E₀⟩ + |E₁₀⟩` |
| **Target gap** | `ΔE₀,₁₀ = 5.9175` |
| **Observables** | 3-local Pauli classical shadows (`k = 3`) |
| **TE-PAI** | `Δ = π/2⁶ = 0.0491`, `M × n_s` = (3000×1) for fig4-1, (10000×1) for fig4-2 |
| **Trotter** | evaluated via the exact noisy density matrix, `shots` = 3000 / 10000 |
| **Step (both methods)** | constant `dt_T = 0.00800` (`N_div = 500`), `θ = 2·dt_T = 0.0160 ≤ Δ` |
| **Time grid** | `t ∈ [0, 4]`, `N_t = 80`, `dt = 0.0506` |
| **Gate noise** | amplitude damping (T₁), `p₁ = 2.5×10⁻⁵`, `p₂ = 2.5×10⁻⁴` |
| **Post-processing** | standardise → Ljung-Box → taper `e^(−0.1 t)` → cross-correlation → SVD → FFT (zero-pad ×4) |
| **Normalisation** | raw / none (shared vertical axis, arbitrary units) |
| **RNG seed** | 80 (representative; behaviour is seed-independent — see the seed sweep) |

### Gate-count / overhead comparison (at `t_max = 4`)

| quantity | TE-PAI | Trotter |
|---|---|---|
| Trotter steps (`δt = 0.008`) | 500 | 500 |
| circuit gates | ≈ 2.4×10³ (`E[ν] = csc Δ (3−cos Δ)‖H‖₁ t`, drops identities) | 7.5×10³ (`J·K = 15·500`) |
| squared sampling overhead `Γ²` | `exp[2 t ‖H‖₁ tan(Δ/2)] ≈ 19` | 1 |

### Result

Both methods use the same `δt` (same Trotter accuracy); TE-PAI is ~3× shallower. Under
amplitude damping the noisy TE-PAI spectrum keeps its dominant peak at `ΔE₀,₁₀`, while the
noisy Trotter spectrum collapses (dominant peak shifts to a spurious low-frequency
decoherence mode ~1.2). Increasing the shot budget from M = 3000 (fig4-1) to M = 10000
(fig4-2) does **not** recover the gap — it sharpens the erroneous low-frequency peak,
showing the bottleneck is gate noise (circuit depth), not the measurement budget.
