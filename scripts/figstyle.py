"""
Shared style for every figure in the manuscript (scripts/make_figures.py).

Rules, so that the figures read as one set:
  - Okabe-Ito colours only (plus greys); no saturated yellow, so the Okabe-Ito yellow is left out; no hatching.
    Where greyscale matters, use marker shape or line style.
  - DejaVu Sans; 8 pt text and tick labels, 9 pt axis labels and panel titles, bold 10 pt panel letters.
  - Final widths are 16 cm (double column) or 8 cm (single column). The figure is created at exactly that size
    and saved without cropping, at 300 dpi, so that 8 pt stays 8 pt in the manuscript.
  - Light y-grid (x-grid for horizontal bars), no top or right spines, legends inside the axes or in one row
    below the axes, never floating.
  - n and what error bars mean go in the caption, not in the image.

Usage:
    import figstyle as fs
    fs.apply()
    fig, axes = fs.new_figure("double", 6.0, 1, 2)
    fs.panel_title(axes[0], "Title of what it shows", letter="a")
    fs.save(fig, "figures/name")
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.transforms import ScaledTranslation

CM = 1.0 / 2.54
WIDTH_CM = {"double": 16.0, "single": 8.0}
DPI = 300

TEXT_PT, LABEL_PT, LETTER_PT = 8, 9, 10

# Okabe & Ito (2008) without the yellow, which is too light on white.
OI = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
}
CYCLE = [OI["blue"], OI["vermillion"], OI["green"], OI["purple"], OI["sky"], OI["orange"], OI["black"]]

# ungrouped validation rungs: three greys, light to dark; grouped rungs: two blues
UNGROUPED_GREYS = ("#D2D2D2", "#A3A3A3", "#6E6E6E")
GROUPED_BLUES = (OI["sky"], OI["blue"])
GRID_COLOR = "0.88"
EDGE_GREY = "0.25"


def apply():
    """Set the project-wide matplotlib rcParams."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": TEXT_PT,
            "axes.labelsize": LABEL_PT,
            "axes.titlesize": LABEL_PT,
            "axes.titleweight": "normal",
            "xtick.labelsize": TEXT_PT,
            "ytick.labelsize": TEXT_PT,
            "legend.fontsize": TEXT_PT,
            "figure.titlesize": LABEL_PT,
            "mathtext.fontset": "dejavusans",
            "mathtext.default": "regular",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.6,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": GRID_COLOR,
            "grid.linewidth": 0.5,
            "axes.axisbelow": True,
            "axes.prop_cycle": cycler(color=CYCLE),
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.major.pad": 2.0,
            "ytick.major.pad": 2.0,
            "lines.linewidth": 1.0,
            "legend.frameon": False,
            "legend.handlelength": 1.4,
            "legend.handletextpad": 0.5,
            "legend.columnspacing": 1.2,
            "savefig.dpi": DPI,
            "figure.dpi": 100,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def width_cm(width):
    """Resolve 'double'/'single' (or a number of cm) to a width in cm."""
    return WIDTH_CM[width] if isinstance(width, str) else float(width)


def new_figure(width="double", height_cm=6.0, nrows=1, ncols=1, **subplot_kw):
    """
    Create a figure of exactly `width` (16 or 8 cm) by `height_cm`, with constrained layout. Returns
    (fig, axes) as plt.subplots does. Extra keyword arguments (width_ratios, gridspec_kw, ...) go to
    plt.subplots.
    """
    fig, axes = plt.subplots(nrows, ncols, figsize=(width_cm(width) * CM, height_cm * CM), layout="constrained", **subplot_kw)
    fig.get_layout_engine().set(w_pad=2 / 72, h_pad=2 / 72, wspace=0.03, hspace=0.03)
    return fig, axes


def panel_title(ax, title, letter=None):
    """
    Title `ax` with what it shows (9 pt, left-aligned). With `letter`, a bold 10 pt panel letter sits at the
    top left and the title starts just to its right.
    """
    t = ax.set_title(title, loc="left", fontsize=LABEL_PT, pad=3)
    if letter is not None:
        ax.annotate(
            letter, xy=(0, 1), xycoords="axes fraction", xytext=(0, 3), textcoords="offset points",
            ha="left", va="baseline", fontsize=LETTER_PT, fontweight="bold",
        )
        t.set_transform(t.get_transform() + ScaledTranslation(0.16, 0, ax.figure.dpi_scale_trans))
    return t


def grid(ax, axis="y"):
    """Light grid on one axis only ('y' for vertical values, 'x' for horizontal bars)."""
    ax.grid(False)
    ax.grid(True, axis=axis, color=GRID_COLOR, linewidth=0.5)
    ax.set_axisbelow(True)


def legend_row(fig, handles, labels, ncol=None):
    """One legend below all axes, in a single row (or as few rows as needed)."""
    ncol = ncol or len(labels)
    return fig.legend(handles, labels, loc="outside lower center", ncol=ncol)


def save(fig, path_without_ext, expect_width_cm=None):
    """
    Save PNG (300 dpi) and PDF with no cropping, and check that the file has the intended physical width,
    so that 8 pt text is 8 pt in the manuscript.
    """
    from PIL import Image

    w_cm = fig.get_size_inches()[0] / CM
    if expect_width_cm is not None:
        assert abs(w_cm - expect_width_cm) < 0.01, (w_cm, expect_width_cm)
    fig.savefig(f"{path_without_ext}.png", dpi=DPI)
    fig.savefig(f"{path_without_ext}.pdf", metadata={"CreationDate": None})  # no timestamp: reproducible PDFs
    with Image.open(f"{path_without_ext}.png") as im:
        got_cm = im.size[0] / DPI * 2.54
    assert abs(got_cm - w_cm) < 0.02, (path_without_ext, got_cm, w_cm)
    return w_cm


def check_text_fits(fig, ax, checks, margin=0.08):
    """
    For schematics drawn in cm-sized data units: every (text, (x0, y0, x1, y1)) pair must lie inside its box,
    and no text may be smaller than 8 pt. Raises AssertionError on the first overflow.
    """
    fig.canvas.draw()
    inv = ax.transData.inverted()
    renderer = fig.canvas.get_renderer()
    for text, (bx0, by0, bx1, by1) in checks:
        assert text.get_fontsize() >= TEXT_PT - 1e-9, (text.get_text()[:30], text.get_fontsize())
        bb = text.get_window_extent(renderer)
        (tx0, ty0), (tx1, ty1) = inv.transform((bb.x0, bb.y0)), inv.transform((bb.x1, bb.y1))
        assert tx0 >= bx0 + margin and tx1 <= bx1 - margin and ty0 >= by0 + margin and ty1 <= by1 - margin, (
            f"text overflows its box: {text.get_text()[:40]!r} "
            f"text=({tx0:.2f},{ty0:.2f},{tx1:.2f},{ty1:.2f}) box=({bx0:.2f},{by0:.2f},{bx1:.2f},{by1:.2f})"
        )
