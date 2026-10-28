"""LaTeX-typeset figure style sized for Springer journals (84 mm / 174 mm columns)."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from cycler import cycler  # noqa: E402

from . import config  # noqa: E402

MM = 1 / 25.4
SINGLE = 84 * MM      # single-column width [in]
DOUBLE = 174 * MM     # double-column width [in]

# Categorical slots (validated reference order); identity is always doubled by marker/line style
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#d9d8d4"

FAMILY_STYLE = {
    "GL24*": dict(color=PALETTE[0], marker="o", ls="-"),
    "GL27*": dict(color=PALETTE[1], marker="s", ls="--"),
    "GL28*": dict(color=PALETTE[2], marker="^", ls="-."),
    "GL33*": dict(color=PALETTE[3], marker="D", ls=":"),
}

# Products: colour follows the family, marker identifies the product
PRODUCT_MARKER = {"P1": "o", "P2": "v", "P3": "^", "P4": "X", "P5": "s", "P6": "P", "P7": "D"}
PRODUCT_FAMILY = {"P1": "GL24*", "P2": "GL24*", "P3": "GL24*", "P4": "GL24*", "P5": "GL27*", "P6": "GL28*",
                  "P7": "GL33*"}


def product_style(p):
    return dict(color=FAMILY_STYLE[PRODUCT_FAMILY[p]]["color"], marker=PRODUCT_MARKER[p])


LABEL = {
    "fm": r"$f_{\mathrm{m}}$ [MPa]",
    "fm600": r"$f_{\mathrm{m}}/k_h$ [MPa]",
    "E": r"$E_{\mathrm{m,g}}$ [GPa]",
    "h": r"$h$ [mm]",
    "b": r"$b$ [mm]",
    "hext": r"$h_{\mathrm{ext}}$ [mm]",
    "hint": r"$h_{\mathrm{int}}$ [mm]",
    "rho": r"$\rho$ [kg/m$^3$]",
}


def use_latex_style():
    plt.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.5,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "axes.prop_cycle": cycler(color=PALETTE),
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.3,
        "axes.axisbelow": True,
        "lines.linewidth": 1.0,
        "lines.markersize": 3.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "xtick.color": INK,
        "ytick.color": INK,
        "legend.frameon": False,
        "legend.handlelength": 1.8,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def figure(width="double", height_ratio=0.42, nrows=1, ncols=1, **kw):
    w = DOUBLE if width == "double" else SINGLE
    fig, ax = plt.subplots(nrows, ncols, figsize=(w, w * height_ratio), constrained_layout=True, **kw)
    return fig, ax


def panel_label(ax, text, x=0.02, y=0.97):
    ax.text(x, y, f"({text})", transform=ax.transAxes, ha="left", va="top", color=INK)


def save(fig, name):
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIGURES / f"{name}.pdf")
    fig.savefig(config.FIGURES / f"{name}.png", dpi=300)
    plt.close(fig)
    return config.FIGURES / f"{name}.pdf"
