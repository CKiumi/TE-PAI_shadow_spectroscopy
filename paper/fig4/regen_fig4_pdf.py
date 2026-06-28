"""Regenerate the paper Figure 4 (paper/fig4/fig4.pdf) from the cached data, matching the
design of fig_TE_PAI_shadow_spectro_Different_Ns.pdf (LaTeX Computer Modern serif, large
fonts, light solid grid, black-edged legend, thick lines).

Two panels: noiseless (left) and with gate noise (right), TE-PAI (red) vs Trotter (blue),
each curve normalised to its own peak. Reads fig4_M4000.npz.

Run:  uv run python paper/fig4/regen_fig4_pdf.py
"""
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 16,
    "axes.labelsize": 26,
    "axes.titlesize": 22,
    "legend.fontsize": 18,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
    "axes.linewidth": 1.3,
    "grid.color": "0.82",
    "grid.linewidth": 0.8,
})

HERE = os.path.dirname(os.path.abspath(__file__))
NPZ = os.path.join(HERE, "fig4_M4000.npz")
STY = {"TE-PAI": dict(color="red", lw=2.2), "Trotter_fine": dict(color="blue", lw=2.0)}
LAB = {"TE-PAI": "TE-PAI", "Trotter_fine": "Trotter"}


def main():
    d = np.load(NPZ)
    dom = list(d["dom"]); dt = float(d["dt"]); M = int(d["M"])
    xmax = max(dom) + 1.5
    fig, axes = plt.subplots(1, 2, figsize=(16.5, 5.4), sharex=True, sharey=True)
    for ax, suf in zip(axes, ["", "_noisy"]):
        noisy = suf == "_noisy"
        for key in ["TE-PAI", "Trotter_fine"]:
            fr = d[key + suf + "_f"]; y = np.abs(d[key + suf + "_s"])
            ax.plot(fr, y / (y.max() or 1.0), label=LAB[key] + (", noisy" if noisy else ""), **STY[key])
        for j, g in enumerate(dom):
            ax.axvline(g, color="0.30", ls=(0, (6, 4)), lw=1.8, zorder=1.5,
                       label=r"energy gaps $\Delta E_{k,0}$" if j == 0 else None)
        ax.grid(True, color="0.82", lw=0.8)
        ax.set_xlim(0, xmax); ax.set_ylim(0, 1.05)
        ax.set_xlabel(r"$E$")   # panel titles removed -- noiseless (left) / noisy (right) go in the caption
        ax.legend(loc="upper right", edgecolor="black", framealpha=1.0, fancybox=False)
    axes[0].set_ylabel(r"$\mathrm{I}(E)$, Arbitrary units")
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.14)   # more horizontal gap between the two panels
    out = os.path.join(HERE, "fig4.pdf")
    fig.savefig(out)
    print(f"saved {out}  (M={M})")
    # also a PNG preview for quick viewing (not the paper deliverable)
    fig.savefig("/tmp/fig4_preview.png", dpi=130)


if __name__ == "__main__":
    main()
