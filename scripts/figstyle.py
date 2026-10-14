"""
Shared style for every figure of the manuscript: scripts/make_figures.py (Paper A figures, two of which the thesis paper reuses) and
thesis_paper/scripts/make_figures_thesis.py both import this one file (the thesis_paper copy was removed, 2026-10-14).

Rules, so that the figures read as one set and are very clear:
  - Okabe-Ito colours only (plus greys); no saturated yellow, so the Okabe-Ito yellow is left out; no hatching.
    Where greyscale matters, use marker shape or line style.
  - Helvetica, with Arial (then Liberation Sans, metric-compatible) where Helvetica is not installed; apply() stops if none of them is
    installed, so that a figure is never drawn in a fallback face by accident. Fonts are embedded as TrueType in PDFs (fonttype 42).
  - 9 pt text and tick labels, 10 pt axis labels, bold 11 pt panel letters; axis labels and tick labels are bold.
  - Axis lines 1.3 pt, ticks 1.3 pt wide and 4 pt long (3 pt and 1.0 pt for minor ticks).
  - Final widths are DOUBLE_COLUMN_CM (16 cm, the text width of the manuscript's Word page, so Word shows a figure 1:1) or
    SINGLE_COLUMN_CM (8.5 cm). They are the only place the widths are set: change them here when a journal's specification is known.
    The figure is created at exactly that size and saved without cropping, at 300 dpi, so that 9 pt stays 9 pt on the page and nothing
    is below 8 pt (MIN_PT, used only for long category labels).
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
DOUBLE_COLUMN_CM = 16.0  # the page text width of the manuscript (A4, 2.5 cm margins); to be set to the journal's specification later
SINGLE_COLUMN_CM = 8.5
WIDTH_CM = {"double": DOUBLE_COLUMN_CM, "single": SINGLE_COLUMN_CM}
DPI = 300

TEXT_PT, LABEL_PT, LETTER_PT = 9, 10, 11
MIN_PT = 8  # the smallest text size, for long category labels only (feature names); everything else is TEXT_PT or larger
FONT_STACK = ("Helvetica", "Arial", "Liberation Sans")  # first one installed is used
AXES_LW, TICK_W, TICK_LEN, MINOR_W, MINOR_LEN = 1.3, 1.3, 4.0, 1.0, 2.5
FONT_USED = None  # set by apply(): the face the figures are drawn in

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


def resolve_font():
    """The first face of FONT_STACK that matplotlib can find in regular and in bold; raises if none is installed."""
    from matplotlib import font_manager as fm

    installed = {}
    for f in fm.fontManager.ttflist:
        installed.setdefault(f.name, set()).add(f.weight)
    for name in FONT_STACK:
        w = installed.get(name, set())
        if any(x <= 500 for x in w) and any(x >= 600 for x in w):
            return name
    raise RuntimeError(f"none of {FONT_STACK} is installed in regular and bold: the figures would be drawn in a fallback face")


def apply():
    """Set the project-wide matplotlib rcParams."""
    global FONT_USED
    FONT_USED = resolve_font()
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [FONT_USED],
            "font.size": TEXT_PT,
            "font.weight": "normal",
            "axes.labelsize": LABEL_PT,
            "axes.labelweight": "bold",
            "axes.titlesize": LABEL_PT,
            "axes.titleweight": "normal",
            "xtick.labelsize": TEXT_PT,
            "ytick.labelsize": TEXT_PT,
            "legend.fontsize": TEXT_PT,
            "figure.titlesize": LABEL_PT,
            "mathtext.fontset": "custom",
            "mathtext.rm": FONT_USED,
            "mathtext.it": f"{FONT_USED}:italic",
            "mathtext.bf": f"{FONT_USED}:bold",
            "mathtext.sf": FONT_USED,
            "mathtext.fallback": "stixsans",
            "mathtext.default": "regular",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": AXES_LW,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": GRID_COLOR,
            "grid.linewidth": 0.5,
            "axes.axisbelow": True,
            "axes.prop_cycle": cycler(color=CYCLE),
            "xtick.major.width": TICK_W,
            "ytick.major.width": TICK_W,
            "xtick.major.size": TICK_LEN,
            "ytick.major.size": TICK_LEN,
            "xtick.minor.width": MINOR_W,
            "ytick.minor.width": MINOR_W,
            "xtick.minor.size": MINOR_LEN,
            "ytick.minor.size": MINOR_LEN,
            "xtick.major.pad": 2.5,
            "ytick.major.pad": 2.5,
            "lines.linewidth": 1.4,
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


def embolden_ticks(fig):
    """Make every tick label of every axes of `fig` bold (matplotlib has no rcParam for it); called by save() after the ticks exist."""
    fig.canvas.draw()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            for lab in axis.get_ticklabels(which="both"):
                lab.set_fontweight("bold")
            axis.get_offset_text().set_fontweight("bold")


def save(fig, path_without_ext, expect_width_cm=None):
    """
    Save PNG (300 dpi) and PDF with no cropping, and check that the file has the intended physical width,
    so that 8 pt text is 8 pt in the manuscript.
    """
    from PIL import Image

    embolden_ticks(fig)
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
    and no text may be smaller than TEXT_PT. Raises AssertionError on the first overflow.
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
