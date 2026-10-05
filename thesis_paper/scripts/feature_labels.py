"""
Readable feature labels for the figures, tables and text of the thesis paper, read from docs/feature_labels.csv (built by make_feature_labels.py). Nothing is typed per plot.
"""

from functools import lru_cache
from pathlib import Path

import pandas as pd

TABLE = Path(__file__).resolve().parents[1] / "docs" / "feature_labels.csv"


@lru_cache(maxsize=1)
def _table():
    d = pd.read_csv(TABLE, encoding="utf-8", keep_default_na=False)
    assert d["column"].is_unique and len(d) == 397
    return dict(zip(d["column"], d["label"]))


def label(column):
    """The label of a feature column, e.g. 'CBFV_dev_thermal_conductivity_(W/(m_K))_' -> 'Elemental thermal conductivity, mean abs. deviation (CBFV)'."""
    return _table()[column]


def wrapped(text, width=27):
    """The label broken into lines of at most `width` characters (no truncation): after the comma that separates the property from the statistic, and inside either part where it is still longer."""
    import textwrap

    if ", " not in text:
        return "\n".join(textwrap.wrap(text, width))
    a, b = text.split(", ", 1)
    return "\n".join(textwrap.wrap(a + ",", width) + textwrap.wrap(b, width))


def two_lines(text, limit=30):
    """The label on two lines when it is longer than `limit` characters: broken after the comma that separates the property from the statistic; no truncation."""
    if len(text) <= limit or ", " not in text:
        return text
    a, b = text.split(", ", 1)
    return a + ",\n" + b
