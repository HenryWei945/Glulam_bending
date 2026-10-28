"""Schematic of the cross-section variables and of the EN 408 four-point bending configuration (Fig. 1)."""

import numpy as np
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle

from glulam import plotting as gp

OUTER = "#d9b98a"
INNER = "#f1e2c4"
EDGE = gp.INK


def dim(ax, p0, p1, text, offset=(0, 0), ha="center", va="center", lw=0.5, fs=7):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="<->", mutation_scale=6, lw=lw, color=EDGE))
    m = ((p0[0] + p1[0]) / 2 + offset[0], (p0[1] + p1[1]) / 2 + offset[1])
    ax.text(*m, text, ha=ha, va=va, fontsize=fs)


def cross_section(ax):
    b, h, te, ti = 1.0, 3.0, 0.42, 0.36
    n_in = int(round((h - 2 * te) / ti))
    ti = (h - 2 * te) / n_in
    y = 0.0
    layers = [("ext", te)] + [("int", ti)] * n_in + [("ext", te)]
    for kind, t in layers:
        ax.add_patch(Rectangle((0, y), b, t, facecolor=OUTER if kind == "ext" else INNER, edgecolor=EDGE, lw=0.5))
        y += t
    dim(ax, (-0.25, 0), (-0.25, h), r"$h$", offset=(-0.12, 0), ha="right")
    dim(ax, (b + 0.15, h - te), (b + 0.15, h), r"$h_{\mathrm{ext}}$", offset=(0.08, 0), ha="left")
    dim(ax, (b + 0.15, h - te - ti), (b + 0.15, h - te), r"$h_{\mathrm{int}}$", offset=(0.08, 0), ha="left")
    dim(ax, (0, -0.3), (b, -0.3), r"$b$", offset=(0, -0.17), va="top")
    ax.annotate("outer lamella\n(grade, thickness)", xy=(0.5, te / 2), xytext=(1.55, 0.35), fontsize=6.5,
                ha="left", va="center", arrowprops=dict(arrowstyle="-", lw=0.4, color=EDGE))
    ax.annotate("inner lamellae", xy=(0.5, h / 2), xytext=(1.55, h / 2), fontsize=6.5, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", lw=0.4, color=EDGE))
    ax.annotate("lamination\nadhesive", xy=(0.75, te), xytext=(1.55, 1.05), fontsize=6.5, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", lw=0.4, color=EDGE))
    ax.set_xlim(-0.75, 2.75)
    ax.set_ylim(-0.75, h + 0.35)
    ax.set_aspect("equal")
    ax.axis("off")


def beam(ax):
    L, H, c = 18.0, 1.0, 6.0
    a = (L - c) / 2
    nl = 6
    for i in range(nl):
        ax.add_patch(Rectangle((-0.6, i * H / nl), L + 1.2, H / nl, facecolor=OUTER if i in (0, nl - 1) else INNER,
                               edgecolor=EDGE, lw=0.4))
    # finger joint in the outer tension lamella
    xs = np.linspace(L / 2 + 0.6, L / 2 + 0.9, 7)
    ys = np.where(np.arange(7) % 2 == 0, 0.0, H / nl)
    ax.plot(xs, ys, color=gp.PALETTE[0], lw=0.8)
    ax.annotate("finger joint", xy=(L / 2 + 0.75, 0.02), xytext=(L / 2 + 2.4, -0.62), fontsize=6.5,
                ha="left", va="center", arrowprops=dict(arrowstyle="-", lw=0.4, color=EDGE))
    for x in (0, L):
        ax.add_patch(Polygon([[x, 0], [x - 0.35, -0.55], [x + 0.35, -0.55]], closed=True, facecolor="#bdbcb6",
                             edgecolor=EDGE, lw=0.5))
        ax.plot([x - 0.6, x + 0.6], [-0.62, -0.62], color=EDGE, lw=0.5)
    for x in (a, a + c):
        ax.add_patch(FancyArrowPatch((x, H + 1.3), (x, H + 0.05), arrowstyle="-|>", mutation_scale=7, lw=0.9,
                                     color=gp.PALETTE[7]))
        ax.text(x + 0.2, H + 0.9, r"$F/2$", fontsize=7, ha="left")
    dim(ax, (0, -1.25), (L, -1.25), r"$\ell$ (span, $11h$--$18h$)", offset=(0, -0.38), va="top")
    dim(ax, (0, H + 1.75), (a, H + 1.75), r"$a$", offset=(0, 0.3))
    dim(ax, (a, H + 1.75), (a + c, H + 1.75), r"$c=6h$", offset=(0, 0.3))
    dim(ax, (L + 1.0, 0), (L + 1.0, H), r"$h$", offset=(0.25, 0), ha="left")
    # deflected shape used for the global modulus of elasticity
    x = np.linspace(0, L, 100)
    w = -0.55 * np.sin(np.pi * x / L)
    ax.plot(x, w - 1.95, color=gp.INK_2, lw=0.6, ls="--")
    ax.text(L / 2, -2.7, r"global deflection $\rightarrow E_{\mathrm{m,g}}$;\quad maximum load $\rightarrow f_{\mathrm{m}}$",
            fontsize=6.5, ha="center", va="top")
    ax.set_xlim(-1.5, L + 2.0)
    ax.set_ylim(-3.2, H + 2.4)
    ax.set_aspect("equal")
    ax.axis("off")


def main():
    gp.use_latex_style()
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(gp.DOUBLE, gp.DOUBLE * 0.33))
    ax1 = fig.add_axes([0.0, 0.0, 0.26, 1.0])
    ax2 = fig.add_axes([0.28, 0.0, 0.72, 1.0])
    cross_section(ax1)
    beam(ax2)
    ax1.text(-0.7, 3.3, "(a)", fontsize=8)
    ax2.text(-1.4, 3.2, "(b)", fontsize=8)
    return gp.save(fig, "fig_setup")


if __name__ == "__main__":
    main()
