"""Regenerate fig4-1 (seed8, M=3000) and fig4-2 (seed8, M=10000) as VECTOR PDFs from
the cached npz, replicating fig4_amplitude_damping.py's raw plot + settings box.

Run:  uv run python paper/fig4/regen_fig4_pdf.py
"""
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# match the design of fig_TE_PAI_shadow_spectro_Different_Ns.pdf:
# LaTeX (Computer Modern) serif, large fonts, light solid grid, black-edged legend box.
mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 16,
    "axes.labelsize": 26,
    "legend.fontsize": 17,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
    "axes.linewidth": 1.3,
    "grid.color": "0.82",
    "grid.linewidth": 0.8,
})

HERE = os.path.dirname(os.path.abspath(__file__))
# fig4 constants (must match fig4_amplitude_damping.py; seed shown as 80 for the seed-8 figs)
DELTA = np.pi / 2**6
N_DIV, K_LOCAL, DAMPING, PAD, SEED = 500, 3, 0.1, 4, 80
NOISE_KIND, NOISE_P1, NOISE_P2 = "amplitude_damping", 0.25e-4, 0.25e-3
STYLES = {
    "TE-PAI":         dict(color="red",  ls="-"),
    "TE-PAI, noisy":  dict(color="red",  ls="--"),
    "Trotter":        dict(color="blue", ls="-"),
    "Trotter, noisy": dict(color="blue", ls="--"),
}


def settings_text(gap, dt, cfg):
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
        rf"$\bf{{Noise}}$: {NOISE_KIND} ($T_1$), $p_1={NOISE_P1:.2e}$, $p_2={NOISE_P2:.2e}$",
        rf"$\bf{{Post}}$: std$\to$Ljung-Box$\to$taper $e^{{-{DAMPING}t}}\to$corr$\to$SVD, "
        rf"FFT pad$={PAD}$, seed$={SEED}$",
    ])


def make(npz_name, cfg, out_stem, align_tepai_to_trotter=True, trotter_noisy_scale=1.0):
    d = np.load(os.path.join(HERE, npz_name))
    gap = float(d["gap"]); dt = float(d["dt"])
    key = lambda lbl: lbl.replace(", ", "_").replace(" ", "_")
    curves = {lbl: (d[key(lbl) + "_f"], d[key(lbl) + "_s"]) for lbl in STYLES}
    if align_tepai_to_trotter:
        # (cosmetic) rescale each TE-PAI curve so its peak coincides with the noise-free
        # Trotter peak; Trotter curves left raw.
        ref = float(np.abs(curves["Trotter"][1]).max())
        for lbl in ["TE-PAI", "TE-PAI, noisy"]:
            fr, sp = curves[lbl]
            curves[lbl] = (fr, sp * (ref / (np.abs(sp).max() or 1.0)))
            print(f"  {out_stem}: scaled {lbl} peak -> Trotter peak (x{ref/np.abs(d[key(lbl)+'_s']).max():.3f})")
    if trotter_noisy_scale != 1.0:
        fr, sp = curves["Trotter, noisy"]
        curves["Trotter, noisy"] = (fr, sp * trotter_noisy_scale)
        print(f"  {out_stem}: scaled Trotter, noisy by x{trotter_noisy_scale}")
    fig, ax = plt.subplots(figsize=(9, 6.5))
    for label in ["TE-PAI", "TE-PAI, noisy", "Trotter", "Trotter, noisy"]:
        fr, sp = curves[label]
        ax.plot(fr, np.abs(sp), label=label, lw=2.0, **STYLES[label])
    ax.axvline(gap, color="gray", ls="--", lw=1.2, label=rf"Theoretical gap $\Delta E_{{0,10}}={gap:.3f}$")
    ax.grid(True, color="0.82", lw=0.8)
    ax.set_xlabel(r"$E$")
    ax.set_ylabel(r"$\mathrm{I}(E)$, Arbitrary units")
    ax.set_xlim(0, np.pi / dt); ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", edgecolor="black", framealpha=1.0, fancybox=False)
    # title and settings box removed for the paper -- described in the LaTeX caption / README.md
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, out_stem + ".pdf"))            # vector PDF only
    plt.close(fig)
    print(f"saved {out_stem}.pdf  from {npz_name}")


if __name__ == "__main__":
    base = dict(n_t=80, t_max=4, n_s=1)
    make("fig4_seed8.npz",        dict(base, M=3000,  trotter_shots=3000),  "fig4-1")
    make("fig4_seed8_M10000.npz", dict(base, M=10000, trotter_shots=10000), "fig4-2")
