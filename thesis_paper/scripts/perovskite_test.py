"""
Structural test for perovskite-type ABX3: a space group alone does not decide it (Pnma is shared by the GdFeO3-type perovskite and by the
needle-like NH4CdCl3-type chain structure; Cmcm is not specific either). A structure is called perovskite-type here only if

  1. its reduced formula is ABX3 (three elements, 1:1:3) and X, the element present three times, is more electronegative than A and B
     (an X more electropositive than both is an anti-perovskite, flagged separately);
  2. for at least one assignment of B among the two cations, every B site has exactly six X neighbours (all within 1.25 times the shortest
     B-X distance of that site) forming an octahedron (every neighbour vector has a partner at 145 degrees or more);
  3. every X atom bridges exactly two B atoms, so the octahedra share corners only;
  4. no two B atoms share more than one X atom, which excludes edge-sharing and face-sharing octahedra (chains and hexagonal perovskites);
  5. the B-X bond graph is three-dimensionally connected (Larsen dimensionality 3), which excludes layered and chain frameworks.

`classify(structure)` returns a dict with the verdict and the reason for a rejection. The test is validated in
na11_validate_perovskite_test.py on known perovskites and known non-perovskites; no threshold is tuned on the screening candidates.
"""

import math
from collections import Counter, defaultdict

import numpy as np
from pymatgen.analysis.dimensionality import get_dimensionality_larsen
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import CutOffDictNN
from pymatgen.core import Element

BOND_FACTOR = 1.25  # B-X neighbours within this multiple of the shortest B-X distance of the site
MIN_OPPOSITE_ANGLE = 145.0  # degrees; every octahedron vertex needs a partner at least this far around the B atom
SEARCH_RADIUS = 5.0  # angstrom, upper bound for the neighbour search


def abx3_elements(structure):
    """(X, cations) if the reduced formula is ABX3 with X present three times, else None."""
    amounts = structure.composition.get_el_amt_dict()
    if len(amounts) != 3:
        return None
    g = math.gcd(*[int(round(v)) for v in amounts.values()])
    red = {k: int(round(v)) // g for k, v in amounts.items()}
    if sorted(red.values()) != [1, 1, 3]:
        return None
    x = next(k for k, v in red.items() if v == 3)
    return x, sorted(k for k in red if k != x)


def _octahedral_neighbours(structure, site_index, x_symbol):
    """The X neighbours of a site within BOND_FACTOR times the shortest X distance, or None if the site is not octahedrally coordinated."""
    site = structure[site_index]
    nbrs = [n for n in structure.get_neighbors(site, SEARCH_RADIUS) if n.specie.symbol == x_symbol]
    if len(nbrs) < 6:
        return None
    nbrs.sort(key=lambda n: n.nn_distance)
    d0 = nbrs[0].nn_distance
    bonded = [n for n in nbrs if n.nn_distance <= BOND_FACTOR * d0]
    if len(bonded) != 6:
        return None
    vecs = np.array([n.coords - site.coords for n in bonded])
    vecs = vecs / np.linalg.norm(vecs, axis=1)[:, None]
    cos = vecs @ vecs.T
    for i in range(6):
        partner = min(cos[i, j] for j in range(6) if j != i)  # most opposite neighbour
        if math.degrees(math.acos(max(-1.0, min(1.0, partner)))) < MIN_OPPOSITE_ANGLE:
            return None
    return bonded, d0


def _corner_sharing_3d(structure, b_symbol, x_symbol):
    """Apply tests 2 to 5 for one assignment of B. Returns (ok, reason, dimensionality)."""
    b_idx = [i for i, s in enumerate(structure) if s.specie.symbol == b_symbol]
    bonds = defaultdict(list)  # X site index -> list of (B site index, B-X distance)
    cutoffs = []
    per_b = {}
    for i in b_idx:
        res = _octahedral_neighbours(structure, i, x_symbol)
        if res is None:
            return False, f"{b_symbol} site {i} is not octahedrally coordinated by {x_symbol}", None
        bonded, d0 = res
        per_b[i] = bonded
        cutoffs.append(BOND_FACTOR * d0)
        for n in bonded:
            bonds[n.index].append((i, tuple(int(k) for k in n.image)))
    x_idx = [i for i, s in enumerate(structure) if s.specie.symbol == x_symbol]
    for j in x_idx:
        if len(bonds[j]) != 2:
            return False, f"an {x_symbol} atom bridges {len(bonds[j])} {b_symbol} atoms, not two", None
    # no pair of B atoms may share two X atoms: pair key = (B index a, B index b, relative lattice shift)
    pair_count = Counter()
    for j, lst in bonds.items():
        (ia, ima), (ib, imb) = lst  # images are those of the X atom relative to each B; their difference is the B-B shift
        shift = tuple(int(a - b) for a, b in zip(ima, imb))
        key = (ia, ib, shift) if (ia, ib, shift) <= (ib, ia, tuple(-s for s in shift)) else (ib, ia, tuple(-s for s in shift))
        pair_count[key] += 1
    if any(c > 1 for c in pair_count.values()):
        return False, "two B atoms share more than one X atom (edge- or face-sharing octahedra)", None
    strat = CutOffDictNN({(b_symbol, x_symbol): max(cutoffs) + 1e-6})
    sg = StructureGraph.from_local_env_strategy(structure, strat)
    dim = int(get_dimensionality_larsen(sg))
    if dim != 3:
        return False, f"the {b_symbol}-{x_symbol} framework is {dim}-dimensional", dim
    return True, "", dim


def classify(structure):
    """Verdict for one pymatgen Structure; see the module docstring for the criteria."""
    out = {"abx3": False, "x": None, "anti_perovskite": False, "b": None, "perovskite": False, "dimensionality": None, "reason": ""}
    el = abx3_elements(structure)
    if el is None:
        out["reason"] = "not ABX3 stoichiometry"
        return out
    x, cations = el
    out["abx3"], out["x"] = True, x
    xe = Element(x).X
    cat_x = [Element(c).X for c in cations]
    if xe < min(cat_x):
        out["anti_perovskite"] = True
        out["reason"] = "the element present three times is more electropositive than both others (anti-perovskite)"
        return out
    if not xe > max(cat_x):
        out["reason"] = "the element present three times is not the most electronegative"
        return out
    reasons = []
    for b in cations:
        ok, why, dim = _corner_sharing_3d(structure, b, x)
        if ok:
            out.update({"perovskite": True, "b": b, "dimensionality": dim, "reason": ""})
            return out
        reasons.append(f"B={b}: {why}")
        out["dimensionality"] = dim if dim is not None else out["dimensionality"]
    out["reason"] = "; ".join(reasons)
    return out
