"""
Shared matplotlib style for every figure in Paper A and Paper B.

Call apply_style() once at the top of any figure-generating script,
before creating any figure or axes -- it sets matplotlib rcParams
globally so every figure in the project shares fonts, sizes, spines,
and color palette without repeating boilerplate per plot. Also
provides get_figsize() for consistent multi-panel sizing,
add_panel_label() for bold a/b/c/d labels, and save_figure() to write
the PNG+PDF pair every figure in the project uses.
"""

import matplotlib as mpl

# DejaVu Sans / Liberation Sans ship with matplotlib and most Linux
# distros -- unlike Arial, no licensing/availability gap across
# collaborators' or reviewers' machines.
FONT_FAMILY = ["DejaVu Sans", "Liberation Sans", "sans-serif"]

AXIS_LABEL_SIZE = 13
TICK_LABEL_SIZE = 11
LEGEND_SIZE = 11
PANEL_LABEL_SIZE = 14  # bold a/b/c/d panel labels

DPI_PNG = 300

# Colorblind-safe categorical palette (Okabe & Ito 2008) -- the same
# 8-color set seaborn ships as its "colorblind" palette. Use for
# discrete series; never matplotlib's default tab10.
COLORBLIND_PALETTE = [
    "#000000",  # black
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
]

# Perceptually uniform sequential colormap for continuous data.
CONTINUOUS_CMAP = "viridis"


def apply_style():
    """
    Set project-wide matplotlib rcParams: font, label/tick/legend
    sizes, thin spines with top/right removed, no gridlines by default,
    the colorblind-safe color cycle, and PDF/PS font embedding as
    editable text (fonttype 42) rather than rasterized outlines.
    """
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": FONT_FAMILY,
            "axes.labelsize": AXIS_LABEL_SIZE,
            "xtick.labelsize": TICK_LABEL_SIZE,
            "ytick.labelsize": TICK_LABEL_SIZE,
            "legend.fontsize": LEGEND_SIZE,
            "axes.titlesize": AXIS_LABEL_SIZE,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "axes.linewidth": 0.8,
            "axes.grid": False,
            "grid.color": "0.85",
            "grid.linewidth": 0.5,
            "axes.prop_cycle": mpl.cycler(color=COLORBLIND_PALETTE),
            "savefig.dpi": DPI_PNG,
            "figure.dpi": 100,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def get_figsize(rows, cols, panel_width=3.3, panel_height=2.8):
    """
    Return an (width, height) figsize in inches for an (rows, cols)
    panel grid. Defaults size each panel to ~3.3in wide, a typical
    single-column journal figure width.
    """
    return (panel_width * cols, panel_height * rows)


def add_panel_label(ax, label, x=-0.15, y=1.05):
    """Add a bold panel label (e.g. "a", "b") at an axes-fraction position, top-left by default."""
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=PANEL_LABEL_SIZE,
        fontweight="bold",
        va="bottom",
        ha="right",
    )


def save_figure(fig, path_without_ext):
    """
    Save `fig` as both a 300 DPI PNG and a vector PDF, given a path
    (str or Path) with no file extension.
    """
    fig.savefig(f"{path_without_ext}.png", dpi=DPI_PNG, bbox_inches="tight")
    fig.savefig(f"{path_without_ext}.pdf", bbox_inches="tight")
