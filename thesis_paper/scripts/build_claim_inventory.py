"""
Build thesis_paper/reports/claim_inventory.csv (and claim_inventory_summary.md) from the supervisor's annotated
manuscript (thesis_paper/source/*.docx) and the committed Paper A artifacts.

One row per number or factual claim (kind CLAIM) and one row per bracketed note the supervisor left in red
(kind ISSUE). Every row has a status:
  PAPER_A         Paper A has a verified value; `verified_value` and `source_paths` are READ FROM THE ARTIFACTS by
                  paper_a_values.py, never typed here
  NEW_ANALYSIS    needs a run or computation that does not exist yet (see the NEW_ANALYSIS registry below)
  LITERATURE      a citation to verify
  DROP_CANDIDATE  not reproducible and not essential
  TEXT_ONLY       wording, definition or design statement with nothing to check against data
Numbers in the output come from the manuscript (`text`, `current_value`), from an artifact (`verified_value`), or are
arithmetic on the manuscript's own numbers and the cleaning constants, computed above the claim list and asserted.

Locations and colours (black = thesis, blue = rewritten, red = issue) are read from the docx itself, and every anchor
must match exactly one place in it, so a mistyped quotation stops the build.

Usage (from the repository root):
    python thesis_paper/scripts/build_claim_inventory.py
"""

import csv
import hashlib
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paper_a_values as pav  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "thesis_paper" / "source" / "Perovskite_Thermoelectric_Manuscript.docx"
SRC_SHA256 = "f12d66404e126de1cb7cc32d8843058988f43dc3777ac5669b17c319223fa2d4"
OUT_CSV = REPO / "thesis_paper" / "reports" / "claim_inventory.csv"
OUT_MD = REPO / "thesis_paper" / "reports" / "claim_inventory_summary.md"
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
STATUSES = ("PAPER_A", "NEW_ANALYSIS", "LITERATURE", "DROP_CANDIDATE", "TEXT_ONLY")

# ---------------------------------------------------------------------------------------------------------------
# NEW_ANALYSIS registry: what must be run, from what, and a rough effort (one person, with Kaggle P100 GPU access
# as for Paper A; "GPU-h" = hours on that GPU; effort includes writing and checking the code and the figures)
# ---------------------------------------------------------------------------------------------------------------
NA = {
    "NA1": ("Nested grouped CV: how optimistic is tuning on the evaluation folds?",
            "snapfix featurized CSV (SHA256 recorded in the Paper A runs); src/nested_cv.py (tuning machinery exists); one frozen "
            "set per target for the comparison; chemistry-cluster fold assignment of Paper A",
            "Re-tune inside each outer training fold (20 trials, 3 inner folds) for 4 targets: about 4 to 8 GPU-h for one repeat of 5 outer "
            "folds, 1 to 2 GPU-days for the 5 repeats Paper A reports; 1 day of coding and checking"),
    "NA2": ("Architecture comparison on the Paper A data: LightGBM, random forest, stacking against XGBoost",
            "snapfix CSV; src/nested_cv.py (--model lightgbm / random_forest already implemented; stacking is not); one tune_once per "
            "model and target; chemistry-cluster rung, 5 repeats x 5 folds. Paper A's old comparison figure holds hand-entered pre-fix "
            "values with no committed run, so nothing can be reused",
            "2 to 3 GPU/CPU days of compute (random forest is CPU-heavy) plus 1 day to add a stacking learner and its tests"),
    "NA3": ("SHAP attribution for S, sigma and kappa, top-feature tables and the coarse source shares",
            "frozen hyperparameters per target; snapfix CSV; scripts/shap_attribution_zT.py generalised from zT to the other three targets "
            "(zT is already done for 25 folds in Paper A); regenerate Figures 10 to 12 with the symbols sigma and kappa",
            "about 0.5 GPU-day plus 1 day for code, tables and figures"),
    "NA4": ("Dataset statistics and counts from the Paper A data",
            "snapfix featurized CSV and the File A cleaned CSV (both local, SHA256-checked): per-property coverage, mean, median, SD and "
            "range (Table 2); rows with zT above 3; rows per temperature bin (800 K and 600 K); mean zT before and after each step; "
            "share of perovskite-family rows (Paper B family labels are already committed)",
            "2 to 3 hours of CPU and a short script with asserted outputs"),
    "NA5": ("Cleaning-step diagnostics",
            "src/data_cleaning.py rerun on the saved File A raw pull (never re-run data_acquisition.py): raw data-point count before range "
            "filtering, mean zT after step 6, number of spikes set to NaN in step 11, effect of the pre-2005 and minimum-property filters. "
            "The rerun must reproduce the committed funnel exactly (the funnel file already holds a verification gate)",
            "about 0.5 day"),
    "NA6": ("Carrier-type classifier",
            "snapfix CSV; sign of S as the label; XGBoost classifier with the chemistry-cluster grouped CV of Paper A; metrics: accuracy, "
            "precision, recall, confusion matrix, class balance; the screening step then needs the override rate",
            "0.5 to 1 day (new classification objective in the nested CV machinery; training is cheap)"),
    "NA7": ("Direct versus derived zT under row-level (random) validation",
            "the same 56,088-row subset and each target's frozen hyperparameters as the committed per-target run; random 5-fold instead of "
            "chemistry-cluster folds; src/direct_vs_derived_zt.py needs a split-strategy option",
            "3 to 6 GPU-h and half a day of code"),
    "NA8": ("MAE and RMSE per target, predicted-versus-measured figure",
            "committed per-row predictions of the ladder chemistry rung (results/ladder_regen_snapfix) and of the direct-vs-derived run; "
            "no model fitting",
            "1 to 2 hours of CPU"),
    "NA9": ("ESTM formula counts and seen/unseen split",
            "data/external/estm.xlsx (local); Paper A's external-validation module; the DOI-overlap check already exists as Paper A's "
            "pass (a); count formulas, and formulas seen in training, if the thesis keeps a seen/unseen split",
            "1 to 2 hours; no fitting"),
    "NA10": ("JARVIS cross-domain evaluation (only if the section is kept)",
             "JARVIS dft_3d perovskite subset (download not in the repo), featurisation, refit with frozen hyperparameters, scoring for every "
             "target the manuscript reports",
             "1 to 2 days"),
    "NA11": ("Materials Project screening of lead-free perovskites",
             "Materials Project API key and query date (neither is in the repo); the four full-data models refit with frozen hyperparameters; "
             "the carrier classifier (NA6); a structural perovskite test (corner-sharing six-fold B-X octahedra, e.g. pymatgen local-environment "
             "analysis) replacing the space-group filter; predictions from 300 to 800 K; ranking, Table 11 with E_hull, S, sigma, kappa, "
             "carrier type; uncertainty statement",
             "3 to 5 days, most of it the structural test and its validation against known perovskites and known impostors"),
    "NA12": ("Literature validation of predictions on perovskite oxides, within 300 to 800 K",
             "digitised or tabulated experimental S and zT for doped SrTiO3, doped CaMnO3, LaCoO3 and others at or below 800 K; predictions "
             "from the refit models; the existing Ca3Co4O9 and CaMnO3 rows are at 1000 to 1100 K, outside the training window",
             "1 to 2 days, mostly finding and checking the experimental values"),
    "NA13": ("Feature-selection experiment (only if Table 3 is kept)",
             "Pearson, LassoCV and mutual-information selection per target, then the chemistry-cluster rung; no code for it exists in the repo",
             "about 1 GPU-day"),
}

# ---------------------------------------------------------------------------------------------------------------
# docx extraction
# ---------------------------------------------------------------------------------------------------------------


class Block:
    """One paragraph or one table row of the source, with plain text and colour spans."""

    def __init__(self, idx, kind, section, loc, text, spans):
        self.idx, self.kind, self.section, self.loc, self.text, self.spans = idx, kind, section, loc, text, spans

    def colour_at(self, pos):
        """'R', 'B' or '' for the character at pos."""
        for a, b, c in self.spans:
            if a <= pos < b:
                return c
        return ""


def _run_parts(p):
    parts = []
    for r in p.iter(W + "r"):
        rp = r.find(W + "rPr")
        col = ""
        if rp is not None:
            c = rp.find(W + "color")
            val = c.get(W + "val") if c is not None else None
            col = {"FF0000": "R", "0000FF": "B"}.get(val, "")
        txt = ""
        for x in r:
            if x.tag == W + "t":
                txt += x.text or ""
            elif x.tag == W + "tab":
                txt += "\t"
            elif x.tag == W + "br":
                txt += " "
        if txt:
            parts.append((txt, col))
    return parts


def _join(parts, offset=0):
    text, spans = "", []
    for txt, col in parts:
        if col:
            spans.append((offset + len(text), offset + len(text) + len(txt), col))
        text += txt
    return text, spans


def extract_blocks():
    """Walk the docx body in order. Returns the list of Blocks."""
    assert hashlib.sha256(SRC.read_bytes()).hexdigest() == SRC_SHA256, "source docx changed"
    with zipfile.ZipFile(SRC) as z:
        root = ET.fromstring(z.read("word/document.xml"))
    body = root.find(W + "body")
    blocks, section, caption, in_summary = [], "Front matter", None, False
    for el in body:
        if el.tag == W + "p":
            parts = _run_parts(el)
            text, spans = _join(parts)
            ps = el.find(W + "pPr")
            style = ""
            if ps is not None and ps.find(W + "pStyle") is not None:
                style = ps.find(W + "pStyle").get(W + "val")
            t = text.strip()
            if not t:
                continue
            if style.startswith("Heading"):
                m = re.match(r"^([0-9]+(?:\.[0-9]+)*|[A-Z])\s", t)
                section = m.group(1) if m else t
                if t.startswith("Summary of flagged issues"):
                    section, in_summary = "Summary", True
                elif t == "References":
                    section = "References"
                caption = None
                continue
            if t in ("Abstract", "Highlights"):
                section = t
                continue
            if t.startswith("Keywords:"):
                section = "Front matter"
            m = re.match(r"^Table (\d+):", t)
            if m:
                caption = f"Table {m.group(1)}"
            loc = f"Section {section}" if section[0].isdigit() else section
            m = re.match(r"^Figure (\d+):", t)
            if m:
                loc = f"Figure {m.group(1)} caption"
            if m or t.startswith("Table "):
                pass
            blocks.append(Block(len(blocks), "p", "Summary" if in_summary else section, loc, text, spans))
        elif el.tag == W + "tbl":
            label = caption or (f"Appendix {section}" if len(section) == 1 else f"Section {section}")
            caption = None
            for ri, tr in enumerate(el.iter(W + "tr")):
                cells, spans, text = [], [], ""
                for ci, tc in enumerate(tr.findall(W + "tc")):
                    parts = []
                    for pp in tc.iter(W + "p"):
                        pr = _run_parts(pp)
                        if pr:
                            parts.extend(([(" ", "")] if parts else []) + pr)
                    ctext, _ = _join(parts)
                    if ci:
                        text += " | "
                    ctext_s, cspans = _join(parts, offset=len(text))
                    text += ctext_s
                    spans.extend(cspans)
                text_stripped = text.rstrip()
                blocks.append(Block(len(blocks), "row", "Summary" if in_summary else section, f"{label}, row {ri}", text_stripped, spans))
    return blocks


ABBREV = ("et al", "e.g", "i.e", "Fig", "Eq", "vs", "approx", "Dr", "ca")


def sentence_around(text, start, end):
    """The sentence containing text[start:end]; abbreviations such as 'et al.' and 'Fig.' do not end a sentence."""
    bounds = [0]
    for m in re.finditer(r"\.\s+(?=[A-Z(\[])", text):
        if any(text[: m.start()].endswith(a) for a in ABBREV):
            continue
        bounds.append(m.end())
    bounds.append(len(text) + 1)
    s = max(x for x in bounds if x <= start)
    e = min(x for x in bounds if x > end) - 1
    out = text[s:e].strip()
    return out if len(out) <= 420 else out[:417] + "..."


def sentence_without_notes(block, start, end):
    """sentence_around on the block's text with the red notes cut out, so a claim's quoted sentence is the thesis text only."""
    red = [(a, b) for a, b, c in block.spans if c == "R"]
    keep = [i for i in range(len(block.text)) if not any(a <= i < b for a, b in red)]
    clean = "".join(block.text[i] for i in keep)
    clean = re.sub(r"\s{2,}", " ", clean)
    if any(a <= start < b for a, b in red):
        return sentence_around(block.text, start, end)
    cstart = sum(1 for i in keep if i < start)
    cend = sum(1 for i in keep if i < end)
    # the whitespace collapse above can only shorten the clean text; recompute the offsets on the collapsed string
    raw = "".join(block.text[i] for i in keep)
    prefix = re.sub(r"\s{2,}", " ", raw[:cstart])
    inner = re.sub(r"\s{2,}", " ", raw[cstart:cend])
    return sentence_around(clean, len(prefix), len(prefix) + len(inner))


# ---------------------------------------------------------------------------------------------------------------
# claim specifications
# ---------------------------------------------------------------------------------------------------------------
ROWS = []


def row(anchor, value, status, keys=(), action="", note="", na="", also=""):
    """Register one CLAIM."""
    assert status in STATUSES, status
    ROWS.append(dict(anchor=anchor, value=value, status=status, keys=list(keys), action=action, note=note, na=na, also=also))


def P(anchor, value, keys, action, note="", also=""):
    """Paper A has a verified value."""
    row(anchor, value, "PAPER_A", keys, action, note, "", also)


def N(anchor, value, na, action, note="", keys=()):
    """New analysis needed."""
    row(anchor, value, "NEW_ANALYSIS", keys, action, note, na)


def L(anchor, value, action, note=""):
    """Literature claim to verify."""
    row(anchor, value, "LITERATURE", (), action, note)


def D(anchor, value, action, note="", na=""):
    """Drop candidate."""
    row(anchor, value, "DROP_CANDIDATE", (), action, note, na)


def T(anchor, value, action, note=""):
    """Text only."""
    row(anchor, value, "TEXT_ONLY", (), action, note)


_TC = pav._constants({"TEMP_MIN_K", "TEMP_MAX_K", "TEMP_BIN_WIDTH_K", "MAD_THRESHOLD"})
N_BINS = (_TC["TEMP_MAX_K"] - _TC["TEMP_MIN_K"]) // _TC["TEMP_BIN_WIDTH_K"] + 1  # temperature bins implied by the cleaning constants
MAD_SIGMA = _TC["MAD_THRESHOLD"] / 1.4826  # an unscaled MAD of 3.5 is this many standard deviations for normal data (1.4826 is the normal-consistency factor)
SCREEN_MISLABEL = 100 * 346 / 550  # percent of nominal ABX3 formulas that fail the space-group test, from the thesis's own counts
assert 20586 - 1442 == 19144 and 346 + 204 == 550

RP = ["rows.S", "rows.sigma", "rows.kappa", "rows.zT"]
CH = ["ladder.chem.S", "ladder.chem.sigma", "ladder.chem.kappa", "ladder.chem.zT"]
RUNGS = lambda t: [f"ladder.rand.{t}", f"ladder.k5.{t}", f"ladder.k10.{t}", f"ladder.comp.{t}", f"ladder.chem.{t}", f"gap.{t}"]  # noqa: E731
HYP = ["hyper.S", "hyper.sigma", "hyper.kappa", "hyper.zT"]
LIT_CHECK = "Check the figure against the cited paper itself, quote the exact value with its conditions, and keep the citation."

# ---- Abstract ----------------------------------------------------------------------------------------------------
P("trained on 184,167 curated experimental records", "184,167", ["funnel.final"] + RP, "Replace the row count with the Paper A cleaned dataset and give the per-target counts; the two datasets are different snapshots and pipeline versions.", "The thesis dataset is not the Paper A snapshot; every dependent number must come from the Paper A runs.")
P("335 composition-based features (MAGPIE, CBFV and temperature)", "335", ["features.full"], "Use the Paper A feature set; Paper A applies no correlation filter to the CBFV block.")
P("Chemistry-grouped cross-validation (GroupKFold by parent chemical system)", "GroupKFold by parent chemical system", ["data.clusters", "data.parent_systems"], "Describe the grouping as Paper A defines it (chemistry cluster: dopants below the threshold removed, near-integer amounts snapped, five repeats of five folds with randomised assignment of the largest clusters). No parent-system run exists in Paper A; do not claim one.", "Decision for the supervisor: adopt the chemistry-cluster definition throughout. The two definitions cross-cut each other (recorded in CLAUDE.md), so thesis numbers cannot be relabelled.")
P("lowered R2 by 0.16–0.29 relative to random splits", "0.16–0.29", ["gap.range", "gap.S", "gap.sigma", "gap.kappa", "gap.zT"], "Replace with the Paper A gaps (random 80/20 minus chemistry cluster).")
P("composition-only ceilings of 0.718, 0.603, 0.778 and 0.697", "0.718, 0.603, 0.778 and 0.697", CH, "Replace with the Paper A chemistry-cluster values; the ordering of the four properties is unchanged.")
P("External validation on the independent ESTM database (zT R2 = 0.670; 0.613 for unseen formulas)", "zT R2 = 0.670; 0.613", ["estm.a", "estm.b", "estm.insupport.b"], "Report Paper A's two strata (source-DOI-disjoint and cluster-disjoint) instead of seen/unseen formulas, and the in-support restriction.", "Paper A's strata are defined differently from the thesis's seen/unseen formulas, so no thesis value maps onto a Paper A value one to one.")
P("supports a realistic ceiling of ≈0.65–0.70 for zT", "≈0.65–0.70", ["ladder.chem.zT", "estm.b", "estm.insupport.b"], "Rewrite: the statement is not supported by Paper A, where the internal ceiling is higher and the cluster-disjoint external score is lower. State both numbers and what each measures.")
P("S2σT/κ (R2 = 0.697 vs 0.460)", "0.697 vs 0.460", ["dvd"], "Replace with the Paper A per-target-hyperparameter run.", "Paper A's committed result supersedes an earlier run that used one hyperparameter set for all four models.")
P("consistent drivers: temperature and d-electron count for zT", "temperature and d-electron count", ["shap.zT.top5"], "Keep temperature as the leading driver; in Paper A d-electron count (NdValence) ranks third, behind a CBFV radius feature.", "Paper A's stored SHAP arrays keep five folds of one repeat; the 25-fold group shares are in the summary file.")
N("melting temperature for κ and polarisability for S", "melting temperature for κ and polarisability for S", "NA3", "Run SHAP for S, sigma and kappa under the Paper A models before keeping these statements.")
N("identified 204 lead-free perovskite-type compounds among 550 nominal ABX3 formulas", "204 of 550", "NA11", "Redo the screening with an explicit structural test; the space-group count is not trustworthy (see issue notes).")
N("BaZrSe3 emerged as the most credible candidate (predicted zT ≈ 0.33 at 800 K)", "0.33 at 800 K", "NA11", "Recompute with the refit Paper A models and report the uncertainty.")

# ---- Highlights --------------------------------------------------------------------------------------------------
P("inflation of 0.16–0.29 from random splits", "0.16–0.29", ["gap.range"], "Replace with the Paper A gaps.", "Repeats the abstract.")
P("XGBoost trained on 184,167 cleaned Starrydata2 records with 335 descriptors", "184,167 / 335", ["funnel.final", "features.full"], "Replace with the Paper A counts.", "Repeats the abstract.")
P("R2 ≈ 0.70 for zT and 0.78 for κ", "0.70 / 0.78", ["ladder.chem.zT", "ladder.chem.kappa"], "Replace with the Paper A ceilings.", "Repeats the abstract.")
P("Independent ESTM test confirms generalisation (zT R2 = 0.670)", "0.670", ["estm.a", "estm.b"], "Rephrase: Paper A reports a loss on external data, larger for chemistries absent from training.", "'Confirms generalisation' is not the finding in Paper A.")
N("Space-group screening finds 204 lead-free perovskites; BaZrSe3 ranks top", "204", "NA11", "Depends on the repeated screening.", "Repeats the abstract.")

# ---- 1 Introduction ----------------------------------------------------------------------------------------------
L("approximately 72% of global primary energy consumption is lost after conversion", "72%", LIT_CHECK, "Supervisor note says verified against the Forman et al. abstract.")
L("zT values as high as ≈2 at optimized compositions", "≈2", LIT_CHECK, "Reference [4] is Biswas et al. 2012.")
L("with zT ≈ 1.0–1.4, dominates near-room-temperature applications", "1.0–1.4", LIT_CHECK, "Reference [5] is Poudel et al. 2008; give the composition and temperature of the value.")
L("doped compositions reaching zT of ≈0.2–0.4", "≈0.2–0.4", LIT_CHECK, "Conflicts with the value in Section 2.2 from the same reference (see issue note).")
L("may require weeks of supercomputer time", "weeks", LIT_CHECK, "Reference [11] is the ShengBTE paper; it may not state a time.")
L("Parse et al. [12] predicted zT with R2 = 0.815", "0.815", LIT_CHECK)
L("Jia et al. [13] achieved R2 ≈ 0.90", "≈ 0.90", LIT_CHECK, "CLAUDE.md records this paper as proposing composition-based CV on Starrydata2.")
L("Sun et al. [14] reported R2 > 0.95 using deep neural networks", "> 0.95", LIT_CHECK, "Unverified (paywalled). CLAUDE.md lists a different Sun et al. paper (TabPFN, Cell Reports Physical Science, 10-fold CV); confirm whether [14] is the same work.")
T("random train–test splits can place temperature-series data from the same material in both training and test sets", "mechanism", "Keep; Paper A quantifies the effect, and its Figure 1 illustrates it.")
L("To our knowledge, no thermoelectric study has quantified, on the same data and model, how much this leakage inflates reported performance across validation strategies", "novelty claim", "Qualify the claim. Ho et al. (2026) compare random, composition-wise and family-wise splits on a 3,879-row subset, so a thermoelectric comparison across split strategies exists; the defensible claim is the five-way, same-model, full-snapshot ladder with a noise-floor anchor.", "Paper A's introduction already handles this; reuse its wording.")
T("(RQ1) what is the realistic prediction ceiling", "RQ1-RQ3", "Keep the three questions; the answers change with the Paper A numbers.")
P("a reproducible eleven-step curation of Starrydata2 into 184,167 experimental records (13,605 formulas, 2,834 chemical systems)", "184,167 / 13,605 / 2,834", ["funnel.final", "data.formulas", "data.parent_systems", "data.clusters"], "Replace all three counts.")
P("row-level splits inflate R2 by +0.163 to +0.288", "+0.163 to +0.288", ["gap.range"], "Replace with the Paper A gap range.")
P("confirmed by external validation on the independent ESTM database", "ESTM confirmation", ["estm.a", "estm.b"], "Reword: external scores are lower than internal ones in every property.")
N("by architecture-independence (XGBoost, LightGBM, random forest, stacking)", "four architectures", "NA2", "Run the comparison on the Paper A data, or delete the claim.")
P("a demonstration that direct zT regression outperforms the component-wise route", "direct beats derived", ["dvd"], "Keep, with the Paper A numbers.")
N("a conclusion that is only visible under grouped validation", "only under grouped validation", "NA7", "Needs the random-split run of both pathways; no committed result tests this.")
N("a structure-aware screening of lead-free perovskites", "structure-aware screening", "NA11", "Depends on the repeated screening and its structural test.")

# ---- 2 Background -----------------------------------------------------------------------------------------------
L("In 1821 Thomas Johann Seebeck discovered", "1821", "Cite the original or a history of thermoelectricity and confirm the year.")
L("optimum carrier concentration of ≈1019–1020 cm−3", "≈1019–1020 cm−3", LIT_CHECK, "Reference [3] is Snyder and Toberer 2008.")
L("Values of t between ≈0.80 and 1.05 generally correspond to perovskite formation", "0.80 to 1.05", LIT_CHECK, "Reference [21]; the commonly quoted window differs between sources, so cite the source of this window.")
L("Bartel et al. [22] proposed a revised tolerance factor τ", "τ", LIT_CHECK)
L("Undoped SrTiO3 has a large Seebeck coefficient (≈ −700 µV K−1 at 300 K)", "≈ −700 µV K−1", LIT_CHECK, "Reference [10].")
L("high thermal conductivity (10–12 W m−1 K−1), giving zT below 0.01", "10–12 W m−1 K−1; zT below 0.01", LIT_CHECK)
L("CaMnO3 exhibits S ≈ −350 µV K−1 but zT < 0.1 because of low σ [23]", "≈ −350 µV K−1; zT < 0.1", LIT_CHECK, "Reference [23] needs page numbers and a DOI (issue note).")
L("raises the carrier concentration and zT up to ≈0.2", "≈0.2", LIT_CHECK)
L("with S of 200–600 µV K−1 but modest zT [24]", "200–600 µV K−1", LIT_CHECK)
L("exhibit lattice thermal conductivity below 0.5 W m−1 K−1", "below 0.5 W m−1 K−1", LIT_CHECK, "Attribution already changed once (issue note).")
L("Doped polycrystalline BiSbSe3 has been reported to reach zT close to 1.0 at 800 K with S ≈ −254 µV K−1", "zT ≈ 1.0 at 800 K; S ≈ −254 µV K−1", "Cite the primary experimental paper (issue note).")
L("Cu3SbS3 and Ag3SbS3 show intrinsically low κ attributed to Sb lone-pair activity", "low κ", LIT_CHECK, "Name error flagged in the issue note.")
L("of one compound can take days to weeks", "days to weeks", LIT_CHECK, "Same unsupported timescale as the Introduction.")
P("Known limitations include digitization uncertainty (typically 2–5%)", "2–5%", ["digitization"], "Cite a source for 2 to 5 percent, or replace it with Paper A's measured digitisation agreement.", "Paper A measures label agreement between two databases; it is not a percentage error.")
L("for over 150,000 inorganic compounds via a REST API [31]", "150,000", LIT_CHECK, "The Materials Project count changes with each release; state the release or date used.")
L("contains 5,205 experimental observations for 880 unique materials [32]", "5,205 / 880", LIT_CHECK, "Reference [32] is Na and Chang 2022.")
L("contains BoltzTraP-computed transport properties for thousands of compounds [33]", "thousands", LIT_CHECK)
T("Tree ensembles (XGBoost, LightGBM, gradient-boosted decision trees) dominate composition-based thermoelectric prediction", "dominate", "Soften or cite a survey.")

# ---- Table 1 (literature summary) ----------------------------------------------------------------------------------
L("Parse et al. [12] | XGBoost | Starrydata2", "23.7K rows; zT R2 0.815; 5-fold CV", LIT_CHECK, "The supervisor's note reports about 23,662 points in the paper.")
L("Jia et al. [13] | GBDT | Starrydata2", "92K rows; zT R2 0.89–0.90; composition CV", LIT_CHECK)
L("Ma & Poon [35] | LightGBM | Compiled TE", "14.1K rows; S R2 0.80; zT R2 0.86; split not stated", LIT_CHECK, "Preprint; CLAUDE.md records it as physics-based feature engineering with no split-strategy comparison.")
L("Sun et al. [14] | DNN | Starrydata2", "~50K rows; zT R2 > 0.95; random split", LIT_CHECK, "Every detail unverified (paywalled).")
L("Barua et al. [36] | XGBoost | Starrydata2", "~160K rows; zT R2 ~0.80", LIT_CHECK, "CLAUDE.md records about 160K rows and three external test sets for this paper.")
L("Wang et al. [37] | Stacking", "5.2K rows; zT R2 0.97; 10-fold CV", LIT_CHECK, "CLAUDE.md records 10-fold CV over 5,226 rows from 1,022 materials with no material grouping described, R2 0.970.")
L("Elavunkel & Padhan [38] | Stacking", "small; S R2 0.99; zT R2 0.92; random split", LIT_CHECK, "'small' needs a row count.")
L("obtained R2 of 0.89–0.90 for zT on 92,000 entries", "0.89–0.90; 92,000", LIT_CHECK, "Repeats Table 1.")
P("this is stricter than standard K-fold but less strict than system-level grouping", "composition CV between random and grouped", ["ladder.rand.zT", "ladder.comp.zT", "ladder.chem.zT"], "Supported by the Paper A ladder; cite it.")
L("(∼50,000 entries, R2 > 0.95 under random split), showing that the combination outperforms either scheme alone", "50,000; R2 > 0.95; combination outperforms", LIT_CHECK, "Paper A's descriptor ablation is the in-house test of this point.")
L("Ma and Poon [35] added pair-interaction descriptors (Miedema mixing enthalpy) and dopant properties", "descriptor claim", LIT_CHECK)
L("Barua et al. [36] trained XGBoost on ≈160,000 data points and reported zT R2 ≈ 0.80", "≈160,000; ≈0.80", LIT_CHECK, "The supervisor could not confirm an external range of 0.67 to 0.80.")
L("Stacking ensembles on small, domain-restricted datasets [37, 38] reported R2 of 0.92–0.99", "0.92–0.99", LIT_CHECK)
P("MAGPIE [39] computes six statistics (minimum, maximum, range, mean, average deviation, mode) of 22 elemental properties, yielding 132 features", "132", ["features.magpie"], "Consistent with Paper A.")
L("K-fold CV provides a nearly unbiased estimate of generalization error [42]", "nearly unbiased", LIT_CHECK)
L("Data leakage from improper validation is now recognized as a widespread cause of over-optimistic results across ML-based science [15]", "widespread", LIT_CHECK)
T("(300, 325, . . . , 800 K)", "20 temperatures", f"Fix the arithmetic: the cleaning constants (300 to 800 K, 25 K bins) give {N_BINS} bins, not 20.", "Computed from the stated bin width.")
L("Roberts et al. [43] showed that grouped or blocked CV", "Roberts et al.", LIT_CHECK)
L("Meredig et al. [16] introduced leave-one-cluster-out CV", "Meredig et al.", LIT_CHECK, "CLAUDE.md lists Meredig 2018 as the origin of the leave-one-cluster-out idea.")
L("Xiong et al. [44] showed that standard K-fold overstates the ability to extrapolate", "Xiong et al.", LIT_CHECK)
L("Li et al. [45] demonstrated systematic degradation of materials ML models under distribution shift", "Li et al.", LIT_CHECK)
L("TreeExplainer computes exact values in polynomial time for tree ensembles [46]", "polynomial time", LIT_CHECK)
L("(1) Validation bias is unquantified: no thermoelectric study compares several splitting strategies on the same data and model", "no study compares", "Qualify as in the Introduction: Ho et al. (2026) compare three split strategies on a small subset.")
L("existing screens rely on DFT-trained models or random-split validation", "existing screens", "Support with citations or delete.")

# ---- 3 Methodology ----------------------------------------------------------------------------------------------
N("together containing about 2.3 million data points", "about 2.3 million", "NA5", "Count the raw data points before range filtering, or delete; the committed funnel starts after range filtering.", keys=["funnel.raw", "raw.counts"])
P("starrydata_curves.csv (measurement points) and starrydata_samples.csv (sample metadata)", "two files", ["raw.counts"], "Name the files of the pinned pull (papers, samples, curves) and its snapshot date.")
P("−1000 ≤ S ≤ 1000 µV K−1, 10 ≤ σ ≤ 107 S m−1, 0.05 ≤ κ ≤ 25 W m−1 K−1 and 0 ≤ zT ≤ 4", "bounds", ["clean.bounds"], "Matches the code; keep.")
L("below the amorphous (Cahill–Pohl) limit [48], respectively", "Cahill–Pohl", LIT_CHECK, "The note says the limit is material-specific.")
L("zT ≤ 4 admits all physically plausible values [3]", "zT ≤ 4", LIT_CHECK)
P("Step 3 restricted temperatures to 300–800 K", "300–800 K, 25 K", ["clean.params", "funnel.step3"], "Matches the code.")
P("removed formulas that could not be parsed by the pymatgen Composition class [49]", "step 5", ["funnel.removed.step5"], "Report the Paper A step counts.", "CLAUDE.md records an open defect: placeholder element tokens pass this step as dummy species.")
P("This removed 6,167 rows (2.5%)", "6,167 (2.5%)", ["funnel.removed.step6"], "Replace.")
N("shifted the mean zT only slightly (0.387 to 0.393)", "0.387 to 0.393", "NA5", "Recompute on the Paper A pipeline or delete.")
P("only 1,780 rows (0.8%) were flagged", "1,780 (0.8%)", ["funnel.removed.step7"], "Replace.")
P("removed 35,110 rows (14.8%)", "35,110 (14.8%)", ["funnel.removed.step8"], "Replace; the step removes a larger share in Paper A.")
P("(0.5 for S, κ and zT; 0.8 for σ", "CV thresholds", ["clean.params"], "Matches the code.")
L("following the robust-outlier recommendation of Leys et al. [50]", "Leys et al.", LIT_CHECK)
P("Step 9 applied a median absolute deviation (MAD) filter, removing values with", "MAD 3.5", ["clean.params", "funnel.removed.step9"], "State that the MAD is unscaled in the code, and give the equivalent for normal data.")
N("A few records (8 of 125,283, <0.01%) have zT > 3.0", "8 of 125,283", "NA4", "Count on the Paper A zT rows.")
L("above the maximum reliably reported for bulk thermoelectrics (≈2.6 [20])", "≈2.6", LIT_CHECK)
P("Step 10 removed formulas with fewer than three distinct temperatures", "three temperatures", ["clean.params", "funnel.removed.step10"], "Matches the code.")
P("17,372 spikes were set to NaN and 520 rows in which all properties became NaN were removed", "17,372; 520", ["funnel.removed.step11"], "Replace the row count; the spike count is not logged (see NA5).", "Only the row count is in the committed funnel.", also="NA5")
D("Additional filters on publication year (pre-2005) and minimum property count (≥2 properties per row) were evaluated but removed no rows", "no rows removed", "Delete: neither filter is part of the Paper A pipeline and no run records the check.", "If kept, NA5 covers it.", na="NA5")
P("The final dataset contains 184,167 rows covering 13,605 unique chemical formulas across 2,834 parent chemical systems", "184,167 / 13,605 / 2,834", ["funnel.final", "data.formulas", "data.parent_systems"], "Replace all three counts.")
N("The mean zT of the retained data is 0.44, higher than the value of 0.393 after Step 6", "0.44 vs 0.393", "NA4", "Recompute both on the Paper A data.")
D("because the subsequent DFT-exclusion and statistical-filtering steps (Steps 7–11) preferentially eliminated lower-quality, lower-zT entries", "causal explanation", "Delete or support with the per-step statistics of NA5; no run tests it.", na="NA5")
N("which explains why the mean (17.3 µV K−1) differs strongly from the median (61.5 µV K−1)", "17.3 / 61.5", "NA4", "Recompute on the Paper A data (Table 2).")
N("S (µV K−1) | 91.4% | 17.3 | 61.5 | 173.9 | −452 to 577", "S row of Table 2", "NA4", "Recompute coverage, mean, median, SD and range.", keys=["coverage.S"])
N("σ (S m−1) | 89.2% | 84,813 | 52,384 | 106,287 | 1,884 to 1.2 × 106", "sigma row of Table 2", "NA4", "Recompute.", keys=["coverage.sigma"])
N("κ (W m−1 K−1) | 63.8% | 2.51 | 2.02 | 1.88 | 0.32 to 12.9", "kappa row of Table 2", "NA4", "Recompute.", keys=["coverage.kappa"])
N("zT | 68.0% | 0.44 | 0.33 | 0.39 | 0 to 3.55", "zT row of Table 2", "NA4", "Recompute.", keys=["coverage.zT"])
P("Distribution of thermoelectric properties in the final dataset (184,167 rows)", "184,167", ["funnel.final"], "Replace the count and regenerate the figure from the Paper A data.")
P("Eleven-step data-cleaning pipeline from raw Starrydata2 (≈2.3 M data points) to the final dataset (184,167 rows)", "≈2.3 M; 184,167", ["funnel.final", "funnel.raw"], "Replace by Paper A's cleaning-funnel figure, which is generated from the committed funnel counts.")
P("i.e. 132 features computed from the chemical formula alone", "132", ["features.magpie"], "Consistent with Paper A.")
P("After removing intra-CBFV features with Pearson correlation |r| > 0.95, 202 CBFV features remained", "202", ["features.cbfv"], "Paper A uses all CBFV features without a correlation filter; state that and drop the filter.")
P("giving 132 + 202 + 1 = 335 descriptors in total", "335", ["features.full"], "Replace with the Paper A feature count.")
D("An initial three-step selection pipeline (Pearson filtering, LassoCV and mutual-information ranking) reduced the features to 25–44 per target", "25–44 features", "Drop the feature-selection experiment and Table 3: Paper A uses the full set and its descriptor ablation addresses the question; no code for this experiment exists.", "If kept, rerun (NA13).", na="NA13")
D("S | 0.649 (25 features) | 0.718 | +0.069", "Table 3, S row", "Drop with Table 3.", na="NA13")
D("σ | 0.579 (44 features) | 0.603 | +0.024", "Table 3, sigma row", "Drop with Table 3.", na="NA13")
D("κ | 0.766 (39 features) | 0.778 | +0.012", "Table 3, kappa row", "Drop with Table 3.", na="NA13")
D("zT | 0.695 (32 features) | 0.697 | +0.002", "Table 3, zT row", "Drop with Table 3.", na="NA13")
P("XGBoost with colsample_bytree = 0.3 performs implicit feature subsampling at every tree", "colsample_bytree = 0.3", HYP + ["hyper.search"], "Replace: the Paper A search keeps colsample_bytree between the stated bounds, so 0.3 is outside it, and every target has its own frozen set.")
N("all of which converged within ±0.02 R2 under GroupKFold validation (Section 4.1)", "within ±0.02", "NA2", "Needs the architecture comparison.")
P("κ coverage is 63.8%", "63.8%", ["coverage.kappa"], "Replace.")
P("median imputation (SimpleImputer), standardization (StandardScaler) and regression (XGBRegressor)", "three-stage pipeline", ["pipeline.preproc"], "Describe the Paper A pipeline: no imputation or scaling for the tree models.")
P("Four models were trained for S (µV K−1), log10σ, log10κ and zT", "log10 for sigma and kappa", ["scale"], "Consistent with Paper A.")
T("the entire pipeline was fitted on the training partition of each fold only", "fold-local preprocessing", "Keep only if true of the rerun; in Paper A there is no fitted preprocessing for the tree models, and the hyperparameters were chosen on all rows.")
P("Key hyperparameters were colsample_bytree = 0.3, learning_rate = 0.01, max_depth = 10, n_estimators = 700 and subsample = 0.8", "hyperparameter values", HYP, "Replace with the four frozen sets.")
P("Hyperparameters were selected per target by randomized search (RandomizedSearchCV, 30 sampled configurations)", "30 configurations", ["hyper.search"], "Replace: Paper A uses an Optuna search with its own number of trials and inner folds.")
L("Using the evaluation folds for tuning can introduce a small optimistic bias [56, 57]", "citations", LIT_CHECK)
N("the effect is expected to be minor given the limited number of configurations", "expected to be minor", "NA1", "Measure it with nested grouped CV, or state that it is unmeasured (as Paper A does).")
P("n_estimators | 500 | 700 | [add]", "500 / 700", ["hyper.S", "hyper.zT", "hyper.search"], "Replace the table by the four frozen sets; add the search space.")
P("learning_rate | 0.05 | 0.01 | [add]", "0.05 / 0.01", ["hyper.S", "hyper.zT", "hyper.search"], "As above.")
P("max_depth | 10 | 10 | [add]", "10 / 10", ["hyper.S", "hyper.zT", "hyper.search"], "As above.")
P("subsample | 0.8 | 0.8 | [add]", "0.8 / 0.8", ["hyper.S", "hyper.zT", "hyper.search"], "As above.")
P("colsample_bytree | 0.3 | 0.3 | [add]", "0.3 / 0.3", ["hyper.S", "hyper.zT", "hyper.search"], "As above.")
N("on balanced classes (51.1% p-type, 48.9% n-type)", "51.1% / 48.9%", "NA6", "Recompute class balance on the Paper A rows.")
P("with 2,834 parent systems, entire chemical families are withheld from training in each fold [43]", "2,834 parent systems", ["data.parent_systems", "data.clusters"], "Replace by the chemistry-cluster counts.", "Parent-system grouping is not what Paper A ran.")
P("after restricting to 300–800 K, 4,584 measurements on 862 formulas", "4,584 / 862", ["estm.n_in_scope"], "Replace the measurement count; the formula count needs NA9.", also="NA9")
N("284 (32.9%) were seen during training and 578 (67.1%) were unseen", "284 / 578", "NA9", "Count against the Paper A training set, or replace by Paper A's two strata.", keys=["estm.a", "estm.b"])
D("As a cross-domain test, the models were also evaluated on DFT-computed perovskite data from JARVIS [33]", "JARVIS test", "Move to the supplement or drop: no JARVIS data, code or result exists in the repo.", "If kept, NA10.", na="NA10")
N("energy above the convex hull Ehull ≤ 0.05 eV", "Ehull ≤ 0.05; 0.1 ≤ Eg ≤ 3.0", "NA11", "Record the query, date and API version.")
N("returning 20,586 lead-free candidates", "20,586", "NA11", "Redo with the dated query.")
N("left 19,144 candidates", "19,144", "NA11", "Redo; the thesis's own counts are arithmetically consistent (asserted above), so the error is not in the subtraction.")
N("550 have ABX3 stoichiometry, but only 204 (37%) adopt perovskite-type space groups", "550 / 204 / 37%", "NA11", "Replace the space-group filter by a structural test and recount.")
N("The remaining 346 ABX3 formulas", "346", "NA11", "Recount.")
N("Anti-perovskites (in which an electropositive metal, rather than the anion, occupies the threefold site; 16 compounds", "16 compounds", "NA11", "Recount.")
N("compounds with Eg > 0.6 eV were removed", "Eg > 0.6 eV", "NA11", "Keep as a stated choice; the PBE-gap caveat is in the Results.")
N("compounds containing Tl, Hg, Cd, As or Be were excluded", "toxic-element list", "NA11", "Keep as a stated choice.")
N("predicted S, σ, κ and zT from 300 to 800 K in 100 K steps", "300 to 800 K, 100 K steps", "NA11", "Rerun with the refit models.")
P("Direct zT prediction was used for ranking because it was more accurate under GroupKFold than the derived S2σT/κ route", "direct more accurate", ["dvd"], "Keep, with the Paper A numbers.")
P("SHAP TreeExplainer [46] was applied to each model on a random subsample of 2,000 rows", "2,000 rows", ["shap.coarse"], "Paper A's zT SHAP uses a larger fixed subsample per fold; describe what is actually run.", "Paper A covers zT only.", also="NA3")

# ---- 4 Results -----------------------------------------------------------------------------------------------------
P("R2 was highest for κ (0.778), followed by S (0.718), zT (0.697) and σ (0.603)", "0.778, 0.718, 0.697, 0.603", CH, "Replace; the order of the four properties is the same.")
N("predictions cluster along the diagonal with larger scatter at extreme values, and high-zT materials are compressed towards the mean", "qualitative description", "NA8", "Check against a figure drawn from the committed per-row predictions.")
P("S | 0.718 | 57.9 µV K−1 | 168,318 | 335", "S row of Table 5", ["ladder.chem.S", "rows.S", "features.full"], "Replace R2, rows and features; MAE needs NA8.", also="NA8")
P("log10σ | 0.603 | 0.246 (log10 S m−1) | 164,201 | 335", "sigma row of Table 5", ["ladder.chem.sigma", "rows.sigma", "features.full"], "As above.", also="NA8")
P("log10κ | 0.778 | 0.105 (log10 W m−1 K−1) | 117,575 | 335", "kappa row of Table 5", ["ladder.chem.kappa", "rows.kappa", "features.full"], "As above.", also="NA8")
P("zT | 0.697 | 0.144 | 125,283 | 335", "zT row of Table 5", ["ladder.chem.zT", "rows.zT", "features.full"], "As above.", also="NA8")
P("fold-level standard deviations were ±0.01–0.03 for all targets", "±0.01–0.03", ["foldsd"], "Replace; the zT fold-level SD is above the stated range.")
P("Random split, 5-fold and 10-fold CV give practically identical R2 (maximum difference 0.003)", "0.003", ["ungrouped.spread.S", "ungrouped.spread.sigma", "ungrouped.spread.kappa", "ungrouped.spread.zT"], "Replace with the largest spread across the four targets.")
P("Composition-level CV gives intermediate values (e.g. zT R2 = 0.814)", "0.814", ["ladder.comp.zT"], "Replace.")
P("random/K-fold (≈0.90) > composition CV (0.81) > GroupKFold (0.70)", "0.90 > 0.81 > 0.70", ["ladder.rand.zT", "ladder.comp.zT", "ladder.chem.zT"], "Replace; the hierarchy is the same.")
P("The inflation ∆R2 (Eq. (8)) ranges from +0.163 for κ to +0.288 for σ", "+0.163 to +0.288", ["gap.range"], "Replace; kappa and sigma remain the smallest and largest.")
T("depending on how strongly each property depends on doping rather than composition", "explanation", "Keep as an interpretation, or support it with Paper A's argument about test-set composition.")
P("S | 0.959 | 0.958 | 0.959 | 0.854 | 0.718 | +0.241", "S row of Table 6", RUNGS("S"), "Replace the row; gap column is random minus chemistry cluster.")
P("σ | 0.891 | 0.892 | 0.894 | 0.755 | 0.603 | +0.288", "sigma row of Table 6", RUNGS("sigma"), "As above.")
P("κ | 0.942 | 0.941 | 0.942 | 0.866 | 0.778 | +0.163", "kappa row of Table 6", RUNGS("kappa"), "As above.")
P("zT | 0.904 | 0.902 | 0.904 | 0.814 | 0.697 | +0.207", "zT row of Table 6", RUNGS("zT"), "As above.")
N("LightGBM, random forest and XGBoost converged to within ±0.02 R2 for every target under GroupKFold", "within ±0.02", "NA2", "Needs the architecture comparison.")
N("A stacking ensemble (XGBoost + LightGBM + random forest with a ridge meta-learner) improved R2 by at most 0.002 (Table 7)", "at most 0.002", "NA2", "Needs the comparison.")
N("S | 0.718 | 0.720 | +0.002", "S row of Table 7", "NA2", "Rerun; the XGBoost column comes from Paper A.", keys=["ladder.chem.S"])
N("σ | 0.603 | 0.602 | −0.001", "sigma row of Table 7", "NA2", "As above.", keys=["ladder.chem.sigma"])
N("κ | 0.778 | 0.778 | 0.000", "kappa row of Table 7", "NA2", "As above.", keys=["ladder.chem.kappa"])
N("zT | 0.697 | 0.699 | +0.002", "zT row of Table 7", "NA2", "As above.", keys=["ladder.chem.zT"])
N("This confirms that the performance ceiling is determined by the information content of the compositional features, not by model architecture", "ceiling set by information content", "NA2", "Supported only after NA2; Paper A's descriptor ablation supports the descriptor side.", keys=["ablation.zT"])
N("achieved 89.45% accuracy under GroupKFold on balanced classes", "89.45%", "NA6", "Rerun on the Paper A rows.")
P("ESTM data (4,584 measurements on 862 formulas, 300–800 K) were not used in model development", "4,584 / 862", ["estm.n_in_scope"], "Replace the count; Paper A records that the external set was scored with models frozen beforehand.", also="NA9")
P("the drops from GroupKFold to ESTM are only −0.040 for κ and −0.027 for zT, and zT still achieves R2 = 0.613 for unseen formulas", "−0.040; −0.027; 0.613", ["estmdrop", "estm.a", "estm.b"], "Replace; the drops are larger in Paper A and 'only' no longer fits.")
P("The largest decrease is for σ (−0.200 overall; R2 = 0.299 for unseen formulas)", "−0.200; 0.299", ["estmdrop", "estm.a", "estm.b", "estm.insupport.a", "estm.insupport.b"], "Replace; sigma is the largest drop in Paper A too, partly because ESTM extends below the training conductivity range.")
P("S | 0.718 | 0.631 | 0.839 | 0.536 | −0.087", "S row of Table 8", ["estm.a", "estm.b", "estmdrop"], "Re-tabulate with Paper A's two strata and the in-support restriction.")
P("σ | 0.603 | 0.403 | 0.637 | 0.299 | −0.200", "sigma row of Table 8", ["estm.a", "estm.b", "estmdrop"], "As above.")
P("κ | 0.778 | 0.738 | 0.860 | 0.656 | −0.040", "kappa row of Table 8", ["estm.a", "estm.b", "estmdrop"], "As above.")
P("zT | 0.697 | 0.670 | 0.762 | 0.613 | −0.027", "zT row of Table 8", ["estm.a", "estm.b", "estmdrop"], "As above.")
P("ESTM external validation for (a) S, (b) log κ, (c) zT and (d) log σ. Blue: formulas seen during training; red: unseen (novel) formulas.", "Figure 9", ["estm.a", "estm.b"], "Regenerate from the committed per-row predictions (estm_predictions_pass{a,b}.npz) for both strata.")
D("Validation against DFT-computed perovskite data from JARVIS gave R2 < 0 for S", "R2 < 0", "Move to the supplement or drop.", "No JARVIS result exists in the repo.", na="NA10")
P("Temperature dominates the zT model (mean |SHAP| = 0.111)", "0.111", ["shap.zT.top5"], "Replace with the Paper A value; temperature stays first.")
P("The second feature, mean NdValence (d-electron count), governs carrier transport in transition-metal compounds", "second feature NdValence", ["shap.zT.top5"], "Reword: in Paper A the second feature is a CBFV radius descriptor and NdValence is third.")
N("Thermal conductivity is dominated by the average melting temperature", "avg melting T", "NA3", "Needs the kappa SHAP run.")
N("Electrical conductivity is driven by temperature and electronegativity descriptors", "T and electronegativity", "NA3", "Needs the sigma SHAP run.")
N("the Seebeck coefficient is dominated by polarizability features", "polarizability", "NA3", "Needs the S SHAP run.")
N("T (0.111) | avg polarizability (7.87)", "Table 9, rank 1", "NA3", "Regenerate the table; the zT column is derivable from Paper A's stored SHAP arrays.", keys=["shap.zT.top5"])
N("mean NdValence (0.027) | mode polarizability (7.31)", "Table 9, rank 2", "NA3", "As above.")
N("mean GS volume (0.016) | dev density (7.27)", "Table 9, rank 3", "NA3", "As above.")
N("dev polarizability (0.015) | dev Gilmor valence (6.98)", "Table 9, rank 4", "NA3", "As above.")
N("avg-dev covalent radius (0.013) | T (6.06)", "Table 9, rank 5", "NA3", "As above.")
N("CBFV features dominate the S (74%) and σ (63%) models", "74%; 63%", "NA3", "Needs the S and sigma SHAP runs; define the share (top 10 features in the thesis, all features in Paper A).")
P("whereas the zT model is dominated by temperature (51%)", "51%", ["shap.coarse"], "Replace; the share is over all features in Paper A, so the numbers are not comparable with a top-10 share.")
P("Direct prediction is more accurate (R2 = 0.697 vs 0.460)", "0.697 vs 0.460", ["dvd"], "Replace.")
P("the weakest component model (σ) adds a large further term", "sigma weakest", ["dvd"], "Consistent: sigma is the weakest component in Paper A.")
N("under random-split validation both pathways appeared to agree (R2 ≈ 0.91)", "≈ 0.91", "NA7", "Needs the random-split run of both pathways.")
P("Direct (features → zT) | 0.697 | 0.144", "Table 10, direct", ["dvd"], "Replace R2; MAE needs NA8.", also="NA8")
P("Derived (S2σT/κ) | 0.460 | 0.187", "Table 10, derived", ["dvd"], "Replace R2; MAE needs NA8.", also="NA8")
N("1,442 containing Pb or radioactive/unstable elements (including several uranium-bearing double perovskites) were removed, leaving 19,144", "1,442; 19,144", "NA11", "Recount.")
N("The structural filter (Section 3.6) yielded 204 perovskite-type compounds", "204", "NA11", "Recount with the structural test.")
N("Stoichiometric matching alone would have mislabelled the majority (63%) of nominal ABX3 compounds as perovskites", "63%", "NA11", f"Recount; the thesis's own counts (346 of 550) give {SCREEN_MISLABEL:.1f} percent.")
N("RbEuCl3 | Pm-3m | 0.454 | metastable | 800 | 0.344", "Table 11 rank 1", "NA11", "Regenerate Table 11.")
N("BaZrSe3 | Pnma | 0.373 | stable | 800 | 0.326", "Table 11 rank 2", "NA11", "As above.")
N("SrZrSe3 | Pnma | 0.167 | metastable | 800 | 0.312", "Table 11 rank 3", "NA11", "As above.")
N("DyCrSe3 | Pnma | 0.117 | metastable | 800 | 0.275", "Table 11 rank 4", "NA11", "As above.")
N("NdLuSe3 | Cmcm | 0.496 | stable | 800 | 0.255", "Table 11 rank 5", "NA11", "As above.")
N("PrLuSe3 | Cmcm | 0.494 | stable | 800 | 0.255", "Table 11 rank 6", "NA11", "As above.")
N("TiGeS3 | Pnma | 0.324 | stable | 800 | 0.253", "Table 11 rank 7", "NA11", "As above.")
N("TbCrSe3 | Pnma | 0.114 | metastable | 800 | 0.247", "Table 11 rank 8", "NA11", "As above.")
N("EuHfS3 | Pnma | 0.129 | stable | 800 | 0.218", "Table 11 rank 9", "NA11", "As above.")
N("TaCuS3 | Pnma | 0.418 | stable | 800 | 0.184", "Table 11 rank 10", "NA11", "As above.")
N("The leading candidate is BaZrSe3 (predicted zT = 0.33 at 800 K)", "0.33 at 800 K", "NA11", "Recompute and re-rank after the structural test.")
L("belongs to the chalcogenide-perovskite family [26]", "family membership", LIT_CHECK, "The supervisor flags a missing citation for experimental thermoelectric studies of this family.")
N("RbEuCl3 ranks marginally higher (0.34) but is metastable (Ehull = 0.038 eV atom−1)", "0.34; 0.038", "NA11", "Recompute.")
N("predicted zT between 0.18 and 0.31", "0.18 to 0.31", "NA11", "Recompute.")
N("All candidates peak at 800 K, consistent with zT rising with temperature in this range", "peak at 800 K", "NA11", "Recompute; a peak at the edge of the window is an extrapolation artefact risk.")
L("Because PBE systematically underestimates band gaps [58]", "PBE underestimates gaps", LIT_CHECK)
N("No lead-free perovskite in the screened set is predicted to exceed zT ≈ 0.35", "≈ 0.35", "NA11", "Recompute.")
N("perovskites are under-represented in Starrydata2, which is dominated by chalcogenide and telluride chemistries", "under-represented", "NA4", "Quantify with the committed Paper B family labels (perovskite-titanate and other oxide rows against the total).")
N("Given the GroupKFold MAE of 0.144 for zT, the prediction for BaZrSe3 (0.33) should be read as an order-of-magnitude estimate (≈0.2–0.5)", "0.144; 0.2–0.5", "NA8", "Recompute the MAE; also reword: an interval of that width is not an order of magnitude.")
N("The model reproduces the zT of Ca3Co4O9 closely (0.111 vs 0.093 at 1000 K)", "0.111 vs 0.093", "NA12", "Outside the training window; redo within 300 to 800 K.")
N("predicts its Seebeck coefficient within ≈6% (+164 vs +175 µV K−1)", "≈6%; +164 vs +175", "NA12", "As above.")
N("(0.072 vs 0.11 for a CaMnO3/CaMn2O4 composite)", "0.072 vs 0.11", "NA12", "As above.")
N("but overestimates the high-temperature Seebeck magnitude", "overestimates", "NA12", "As above.")
N("CaMnO3 | composite | 1100 | 0.072 | 0.11 | −286 | ≈−150* | n | [59]", "Table 12, CaMnO3", "NA12", "Predictions need rerunning; experimental values need checking against [59].")
N("Ca3Co4O9 (x = 0) | single-phase | 1000 | 0.111 | 0.093 | +164 | +175 | p | [60]", "Table 12, Ca3Co4O9", "NA12", "As above, against [60]; the compound is not a perovskite.")
P("Under random-split validation, the Seebeck model (R2 = 0.959) is comparable to or better than published results", "0.959", ["ladder.rand.S"], "Replace.")
P("Under composition-level CV (the strategy of Jia et al. [13]), the zT model achieves R2 = 0.814, compared with their 0.89–0.90", "0.814", ["ladder.comp.zT"], "Replace; keep the comparison to Jia et al. as literature.")
P("(184,167 vs 92,000 rows)", "184,167", ["funnel.final", "rows.zT"], "Replace the thesis count.")
T("and stricter cleaning, which removes easy-to-predict duplicate entries", "explanation", "Delete or label as a hypothesis: no run supports it.")
P("Under GroupKFold, zT drops further to R2 = 0.697", "0.697", ["ladder.chem.zT"], "Replace.")
L("is consistent with the R2 ≈ 0.80 reported by Barua et al. [36] on a similar-sized dataset without chemistry-grouped splitting", "≈ 0.80", LIT_CHECK, "The comparison is not like for like (different split, different data); also 0.67 and 0.80 are not 'consistent'.")
P("convergent evidence that the composition-only ceiling for zT is ≈0.65–0.70", "≈0.65–0.70", ["ladder.chem.zT", "estm.b"], "Rewrite with the Paper A internal ceiling and the external scores, stating what each measures.")
P("This work | XGBoost | 0.959 | 0.904 | 168K (S) / 125K (zT) | Random split", "Table 13, this work random", ["ladder.rand.S", "ladder.rand.zT", "rows.S", "rows.zT"], "Replace.")
P("This work | XGBoost | 0.718 | 0.697 | 168K (S) / 125K (zT) | GroupKFold", "Table 13, this work grouped", ["ladder.chem.S", "ladder.chem.zT", "rows.S", "rows.zT"], "Replace.")
L("Sun et al. [14] | DNN | N/R", "Table 13, Sun", LIT_CHECK, "Unverified.")
L("Jia et al. [13] | GBDT | N/R | 0.90 | 92K | Comp. CV", "Table 13, Jia", LIT_CHECK, "Table 1 says 0.89 to 0.90.")
L("Parse et al. [12] | XGBoost | N/R | 0.815 | 23.7K | 5-fold CV", "Table 13, Parse", LIT_CHECK)
L("Barua et al. [36] | XGBoost | N/R", "Table 13, Barua", LIT_CHECK, "Validation scheme marked 'verify'.")
L("Ma & Poon [35] | LightGBM | 0.80 | 0.86", "Table 13, Ma and Poon", LIT_CHECK)
P("Prediction of electrical conductivity is intrinsically limited (GroupKFold R2 = 0.603; ESTM unseen R2 = 0.299)", "0.603; 0.299", ["ladder.chem.sigma", "estm.a", "estm.b"], "Replace both numbers.")
T("because carrier concentration depends on doping and synthesis conditions not reflected in the formula", "explanation", "Keep as the standard explanation, with a citation.")
N("The carrier-type classifier (89.45% accuracy) mitigates sign errors in S, but about one in ten materials could still be assigned the wrong sign", "89.45%; one in ten", "NA6", "Recompute.")
D("Materials Project structural features (formation energy, band gap, density) were tested but covered only 4.5% of formulas and degraded performance", "4.5%", "Delete: no run or code for this test exists in the repo.", "If kept, it needs a new experiment (not itemised).")
N("training range where data are sparsest (2,502 rows)", "2,502 rows", "NA4", "Count rows in the 800 K bin of the Paper A data.")
N("re-ranking at 600 K, where the training data are densest", "600 K densest", "NA4", "Check the row counts per bin.")
L("ensemble variance or conformal prediction [61] would make the screening more decision-ready", "conformal prediction", LIT_CHECK)

# ---- 5 Conclusions ---------------------------------------------------------------------------------------------------
P("using 184,167 curated experimental records from Starrydata2 and 335 MAGPIE, CBFV and temperature descriptors", "184,167 / 335", ["funnel.final", "features.full"] + RP, "Replace with the Paper A counts.", "Repeats the abstract.")
P("chemistry-grouped GroupKFold reduces R2 by 0.163 (κ) to 0.288 (σ). (RQ2)", "0.163 to 0.288", ["gap.range"], "Replace.")
P("the composition-only ceilings are R2 = 0.718 (S), 0.603 (σ), 0.778 (κ) and 0.697 (zT)", "0.718, 0.603, 0.778, 0.697", CH, "Replace.")
N("XGBoost, LightGBM, random forest and stacking converge within ±0.02, so the bottleneck is feature information content, not model complexity. (RQ1)", "within ±0.02", "NA2", "Needs NA2.")
P("External validation on ESTM confirms generalization (zT R2 = 0.670, only −0.027 below GroupKFold; 0.613 on unseen formulas), while σ remains poorly predictable (R2 = 0.299 for unseen formulas)", "0.670; −0.027; 0.613; 0.299", ["estmdrop", "estm.a", "estm.b"], "Replace and reword.")
D("experimental-to-DFT transfer fails (JARVIS R2 < 0)", "JARVIS R2 < 0", "Drop with the JARVIS test.", na="NA10")
P("Direct zT regression (R2 = 0.697) outperforms the component-wise route S2σT/κ (R2 = 0.460) because component errors compound", "0.697; 0.460", ["dvd"], "Replace.")
N("SHAP analysis confirms physically meaningful learning: temperature and d-electron count for zT, melting temperature for κ, polarizability for S and electronegativity for σ", "SHAP drivers", "NA3", "Needs NA3 for three of four targets.", keys=["shap.zT.top5"])
N("Only 204 of 550 lead-free ABX3 compounds adopt perovskite-type space groups", "204 of 550", "NA11", "Redo.")
N("no screened perovskite exceeds zT ≈ 0.35", "≈ 0.35", "NA11", "Redo.")
D("which could raise structural coverage from 4.5% to an estimated 60–70%", "4.5% to 60–70%", "Delete the estimate: it has no basis in a committed result.")
T("(Bi0.5Sb1.5Te3 → Bi2Te3)", "example", "Keep.")

# ---- Declarations, references -----------------------------------------------------------------------------------------
P("The raw data are publicly available from Starrydata2, the Materials Project and ESTM.", "public data", ["raw.counts"], "State the pinned snapshot and its hash: the live Starrydata2 export changes daily and the Paper A snapshot is archived separately.")
T("the authors used Claude (Anthropic) to restructure the thesis into journal-article format and to improve language", "AI tools used", "List every use, including analysis code, as the Paper A statement does.")
L("Data-driven analysis of electron relaxation times in", "ref [30]", "The thesis cites this 2019 paper as the Starrydata2 reference; Paper A cites the Starrydata database paper (Katsura et al. 2025). Decide which is correct and use one.")
L("Rationally design thermoelectric materials based on ingenious machine learning methods", "ref [14]", "Check whether this is the work with the quoted numbers (see the Sun et al. notes).")

# ---------------------------------------------------------------------------------------------------------------
# issue notes: status, proposed resolution, optional keys, NEW_ANALYSIS id
# ---------------------------------------------------------------------------------------------------------------
RES = [
    ("Black text = original thesis material", "TEXT_ONLY", "Delete before submission; it is a legend for the annotated draft.", [], ""),
    ("Corrected: the thesis states", "LITERATURE", "Keep the corrected value and its citation; delete the note.", [], ""),
    ("the thesis cites Li et al. (Cs", "LITERATURE", "Either cite a perovskite band-engineering reference for the claim or delete 'band-structure tuning'.", [], ""),
    ("Section 2.2 of the thesis quotes", "LITERATURE", "Read Ohta et al. 2005, quote one value with its temperature and sample, and use it in both sections.", [], ""),
    ("Sun et al. details unverified", "LITERATURE", "Obtain the paper through the library; confirm rows, split and R2, or remove the figure. Check whether it is the same work as the Sun et al. paper Paper A cites.", [], ""),
    ("the thesis repeatedly claims to be", "LITERATURE", "Use 'to our knowledge' and restrict to thermoelectrics, as Paper A's introduction does; cite Meredig, Xiong and Li.", [], ""),
    ("RQ2 in thesis Chapter 1", "TEXT_ONLY", "Keep the three questions of the Results chapter; state screening as an application. Rewrite the thesis Chapter 1 version in the same terms.", [], ""),
    ("the thesis attributes the", "LITERATURE", "Find a primary source for the low-kappa value; remove the 'theoretical zT above 1' claim unless a theory paper supports it.", [], ""),
    ("(i) BiSbSe", "LITERATURE", "Rename the paragraph (BiSbSe3 is not a perovskite), cite the primary BiSbSe3 experiment, and write Cu3SbS4 for famatinite or correct the compound.", [], ""),
    ("we employ the widely used BoltzTraP code", "TEXT_ONLY", "Already reworded; confirm and delete the note.", [], ""),
    ("the thesis expands ESTM as", "LITERATURE", "Use the name and counts from Na and Chang 2022 and cite it; Paper A already cites that paper.", [], ""),
    ("==[unverified]", "LITERATURE", "Verify against the full paper (Sun et al.), or remove the entry.", [], ""),
    ("==[verify]", "LITERATURE", "Verify the Barua et al. validation scheme in the paper.", [], ""),
    ("Hallucination check", "LITERATURE", "Re-read each paper (Parse, Barua, Wang, Ma and Poon, Sun) and enter the quoted values, including the validation scheme; the wrong-table-caption problem is already fixed.", [], ""),
    ("in the thesis this row cited Wang", "LITERATURE", "Cite the stacking paper (Wang, Zhong, Zhang, Yao et al.), not the review; confirm the table captions now match their tables.", [], ""),
    ("The thesis states external validation with R", "LITERATURE", "Read Barua et al. and quote only what its external sets show.", [], ""),
    ("the thesis version of this figure (and of Fig. 5)", "PAPER_A", "Replace Figure 5 with Paper A's cleaning funnel (generated from the committed counts) and Figure 4 with Paper A's study overview, or draw a new workflow from the final numbers; state in the AI declaration how the figures were made.", ["funnel.final"], ""),
    ("the text gives", "PAPER_A", "The code uses the stated lower bound; keep it in the text and drop the old figure. Describe the Cahill-Pohl bound as permissive, as the note says.", ["clean.bounds"], ""),
    ("Fig. 5 says", "PAPER_A", "Use the exact Paper A step-7 count.", ["funnel.removed.step7"], ""),
    ("state whether MAD was scaled", "PAPER_A", f"State that the code uses the unscaled MAD, so the threshold is {_TC['MAD_THRESHOLD']} unscaled MADs (about {MAD_SIGMA:.1f} standard deviations for normal data).", ["clean.params"], ""),
    ("the thesis cites Dunn et al.", "LITERATURE", "Keep Ward et al. 2018 (already added; Paper A cites it too); delete the Dunn et al. reference unless Matbench was used.", [], ""),
    ("(i) the thesis refers to search", "PAPER_A", "Report the Paper A search space; run nested grouped CV, or state that the optimism of tuning on the evaluation folds was not measured (as Paper A does).", ["hyper.search"], "NA1"),
    ("==[add]", "PAPER_A", "Fill from the Paper A search space.", ["hyper.search"], ""),
    ("Confirm that max_depth, subsample", "PAPER_A", "They differ between targets in the frozen sets: replace Table 4 by the four frozen sets, one column per target.", HYP, ""),
    ("ESTM and Starrydata2 are both digitized", "PAPER_A", "Report Paper A's source-DOI-disjoint stratum alongside the cluster-disjoint one, with the number of overlapping DOIs.", ["estm.a", "estm.b"], ""),
    ("specify the unit of E_hull", "TEXT_ONLY", "Write eV per atom.", [], ""),
    ("report the number of candidates remaining", "NEW_ANALYSIS", "Report the counts after each filter once the screening is rerun.", [], "NA11"),
    ("Critical technical issue", "NEW_ANALYSIS", "Add the connectivity test and rerun the screening; do not keep the space-group count or Table 11 until then. Validate the test on known perovskites and known impostors (needle-like Pnma chalcogenides).", [], "NA11"),
    ("Report the individual fold standard deviation", "PAPER_A", "Fold SDs come from the Paper A runs; RMSE needs the committed predictions.", ["foldsd"], "NA8"),
    ("Fig. 8 shows random-split", "PAPER_A", "Regenerate Table 6, Figure 8 and the text from the Paper A metrics file; the old values disappear.", RUNGS("sigma"), ""),
    ("The thesis text referred to this figure", "TEXT_ONLY", "Already corrected; delete the note.", [], ""),
    ("Add the LightGBM and random-forest", "NEW_ANALYSIS", "Run the architecture comparison on the Paper A data and add the two columns.", [], "NA2"),
    ("Report precision/recall or a confusion matrix", "NEW_ANALYSIS", "Add the metrics from the classifier run and the override rate from the screening.", [], "NA6"),
    ("Fig. 9 reports S seen", "PAPER_A", "Regenerate Figure 9 and Table 8 from one source, the committed ESTM per-row predictions.", ["estm.a", "estm.b"], ""),
    ("Give the number of JARVIS compounds", "DROP_CANDIDATE", "Drop the JARVIS result, or run it and move it to the supplement.", [], "NA10"),
    ("Remove internal labels such as", "NEW_ANALYSIS", "Regenerate the SHAP figures with symbols (part of the SHAP rerun).", [], "NA3"),
    ("the thesis states that", "PAPER_A", "For zT use Paper A's group shares (stating that they are over all features); the other targets need the SHAP rerun. Prefer a stacked bar to pie charts.", ["shap.coarse"], "NA3"),
    ("Add E_hull values, predicted S", "NEW_ANALYSIS", "Add the columns to the regenerated Table 11, and discuss the limiting property.", [], "NA11"),
    ("Eu 4f states are poorly described", "LITERATURE", "Cite the PBE limitation for f-electron systems and flag or drop RbEuCl3; add a citation for experimental studies of the BaZrSe3 family or rephrase.", [], "NA11"),
    ("(i) both comparisons are at", "NEW_ANALYSIS", "Redo the literature check within 300 to 800 K, with perovskite examples; state any extrapolation. Drop Ca3Co4O9 from a perovskite section.", [], "NA12"),
    ("the thesis listed 168K rows", "PAPER_A", "Use the Paper A per-target counts.", RP, ""),
    ("Adjust to reflect all AI tools", "TEXT_ONLY", "List every tool, including the code agent used for the analysis, and how the original Figures 4 and 5 were made.", [], ""),
    ("The chapter first appeared in the 1995 edition", "LITERATURE", "Cite the edition actually consulted.", [], ""),
    ("Add page numbers and DOI", "LITERATURE", "Look up the proceedings paper and add pages and a DOI.", [], ""),
    ("Preprint (not peer reviewed)", "LITERATURE", "Cite the arXiv version with its number, re-read the two R2 values it reports, and say it is a preprint; CLAUDE.md already records that it does not compare split strategies.", [], ""),
    ("Thesis cites 2024; volume 17", "LITERATURE", "Cite the year of the issue (2025), as Paper A does.", [], ""),
    ("insert Dr. Kamran Javed", "TEXT_ONLY", "Insert the e-mail address (author to supply).", [], ""),
]


# ---------------------------------------------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------------------------------------------

def norm(s):
    return re.sub(r"\s+", " ", s).strip()


def red_notes(block):
    """Bracketed red notes (or the whole red text if it has no brackets) in a block, with start offsets."""
    segs = []
    for a, b, c in block.spans:
        if c != "R":
            continue
        if segs and a - segs[-1][1] <= 1:
            segs[-1][1] = b
        else:
            segs.append([a, b])
    notes = []
    for a, b in segs:
        t = block.text[a:b]
        depth, start = 0, None
        found = []
        for i, ch in enumerate(t):
            if ch == "[":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "]" and depth:
                depth -= 1
                if depth == 0:
                    found.append((a + start, t[start:i + 1]))
        if not found:
            found = [(a, t.strip())]
        notes.extend(found)
    return notes


def main():
    """Entry point."""
    blocks = extract_blocks()
    claim_blocks = [b for b in blocks if b.section != "Summary"]

    # ---- claims
    claims, errs = [], []
    for spec in ROWS:
        hits = [(b, b.text.find(spec["anchor"])) for b in claim_blocks if spec["anchor"] in b.text]
        if len(hits) != 1:
            errs.append(f"{len(hits)} matches: {spec['anchor'][:100]!r}" + ("  in " + ", ".join(h[0].loc for h in hits) if hits else ""))
            continue
        b, pos = hits[0]
        vpos = b.text.find(spec["value"], pos)
        vpos = vpos if 0 <= vpos < pos + len(spec["anchor"]) else pos
        col = b.colour_at(vpos)
        origin = {"B": "rewritten (blue)", "R": "red note", "": "thesis (black)"}[col]
        vals, srcs = [], []
        for k in spec["keys"]:
            v, s = pav.get(k)
            vals.append(f"{k}: {v}")
            srcs.extend(s)
        text = b.text if b.kind == "row" else sentence_without_notes(b, pos, pos + len(spec["anchor"]))
        claims.append(dict(kind="CLAIM", order=(b.idx, pos), location=b.loc, text=text, current_value=spec["value"], origin=origin,
                           status=spec["status"], verified_value=" || ".join(vals), source_paths="; ".join(dict.fromkeys(srcs)),
                           action=spec["action"], notes=spec["note"], na=spec["na"] or "", also=spec["also"]))

    if errs:
        print(chr(10).join(errs))
        sys.exit(f"{len(errs)} anchors failed")

    # ---- issue notes
    issues, used = [], Counter()
    for b in claim_blocks:
        for off, note in red_notes(b):
            body = note.strip()
            inner = body[1:-1] if body.startswith("[") and body.endswith("]") else body
            matched = None
            for i, (key, status, res, keys, na) in enumerate(RES):
                if key.startswith("=="):
                    ok = body == key[2:]
                else:
                    ok = key in inner
                if ok:
                    assert matched is None, f"note matches two resolutions: {inner[:60]!r}"
                    matched = i
            if matched is None:
                errs.append(f"unresolved note: {inner[:110]!r}")
                continue
            used[matched] += 1
            key, status, res, keys, na = RES[matched]
            vals, srcs = [], []
            for k in keys:
                v, s = pav.get(k)
                vals.append(f"{k}: {v}")
                srcs.extend(s)
            issues.append(dict(kind="ISSUE", order=(b.idx, off), location=b.loc, text=body, current_value="", origin="red note",
                               status=status, verified_value=" || ".join(vals), source_paths="; ".join(dict.fromkeys(srcs)),
                               action=res, notes="", na=na, also=""))
    # summary-only items: the e-mail request has no body counterpart
    summary_items = [norm(b.text) for b in blocks if b.section == "Summary" and any(c == "R" for _, _, c in b.spans)]
    email = [t for t in summary_items if "institutional e-mail" in t]
    assert len(email) == 1
    for i, (key, *_r) in enumerate(RES):
        if "Kamran" in key:
            used[i] += 1
            key, status, res, keys, na = RES[i]
            issues.append(dict(kind="ISSUE", order=(10**6, 0), location="Summary list only (no body location)", text=email[0], current_value="", origin="red note",
                               status=status, verified_value="", source_paths="", action=res, notes="", na=na, also=""))
    if errs:
        print(chr(10).join(errs))
        sys.exit(f"{len(errs)} notes unresolved")
    unused = [RES[i][0] for i in range(len(RES)) if not used[i]]
    assert not unused, f"resolution keys never used: {unused}"

    # reconcile with the summary list: every substantive body note should appear in the summary
    body_norm = [norm(i["text"].strip("[]")) for i in issues if len(i["text"]) > 30 and "institutional" not in i["text"]]
    summ_blob = " ".join(summary_items)
    missing_in_summary = [t[:70] for t in body_norm if t[:60] not in summ_blob]

    allrows = sorted(claims + issues, key=lambda r: r["order"])
    ids = Counter()
    for r in allrows:
        ids[r["kind"]] += 1
        r["id"] = f"{'C' if r['kind'] == 'CLAIM' else 'I'}{ids[r['kind']]:03d}"
    na_lookup = defaultdict(list)
    for r in allrows:
        for n in filter(None, [r["na"], r["also"]]):
            na_lookup[n].append(r["id"])
    assert set(na_lookup) <= set(NA), set(na_lookup) - set(NA)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    cols = ["id", "kind", "location", "text", "current_value", "origin", "status", "verified_value", "source_paths", "action", "new_analysis_id", "also_needs", "notes"]
    with open(OUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(cols)
        for r in allrows:
            w.writerow([r["id"], r["kind"], r["location"], r["text"], r["current_value"], r["origin"], r["status"], r["verified_value"],
                        r["source_paths"], r["action"], r["na"], r["also"], r["notes"]])

    # ---- summary
    c_claim = Counter(r["status"] for r in allrows if r["kind"] == "CLAIM")
    c_issue = Counter(r["status"] for r in allrows if r["kind"] == "ISSUE")
    tags = sum(1 for i in issues if i["text"] in ("[add]", "[unverified]", "[verify]"))
    lines = ["# Claim inventory: summary", "",
             "Generated by `thesis_paper/scripts/build_claim_inventory.py` from `thesis_paper/source/Perovskite_Thermoelectric_Manuscript.docx` "
             f"(SHA256 `{SRC_SHA256}`). Rows: `thesis_paper/reports/claim_inventory.csv`.", "",
             "## Counts per status", "", "| Status | Claims | Issue notes | Total |", "|---|---:|---:|---:|"]
    for s in STATUSES:
        lines.append(f"| {s} | {c_claim[s]} | {c_issue[s]} | {c_claim[s] + c_issue[s]} |")
    lines.append(f"| **All** | **{sum(c_claim.values())}** | **{sum(c_issue.values())}** | **{sum(c_claim.values()) + sum(c_issue.values())}** |")
    lines += ["", f"The source holds {len(issues) - 2} bracketed or red notes in the body ({tags} of them one-word tags such as [add], [unverified] and "
              f"[verify]), one red legend, and a 'Summary of flagged issues' list that repeats them; one item of that list (the e-mail address) has no body location. "
              f"All {len(issues)} are in the CSV as ISSUE rows, each with a proposed resolution. The task description says 50; the document contains the number above.",
              ""]
    if missing_in_summary:
        lines += ["Body notes not found in the summary list (informational): " + "; ".join(f"`{m}`" for m in missing_in_summary), ""]
    lines += ["## NEW_ANALYSIS items", "",
              "Each item lists the inventory rows it feeds. Effort is a rough single-person estimate with the Kaggle P100 used for Paper A.", ""]
    for n_id in sorted(NA, key=lambda s: int(s[2:])):
        title, inputs, effort = NA[n_id]
        rows = na_lookup.get(n_id, [])
        lines += [f"### {n_id}. {title}", "", f"- **Feeds**: {len(rows)} rows ({', '.join(rows)})", f"- **Inputs**: {inputs}", f"- **Effort**: {effort}", ""]
    lines += ["## DROP_CANDIDATE rows", ""]
    for r in allrows:
        if r["status"] == "DROP_CANDIDATE":
            lines.append(f"- {r['id']} ({r['location']}): {r['text'][:140]}")
    lines += ["", "## Decisions that change the inventory", "",
              "1. Grouping definition. The thesis groups by parent chemical system; Paper A groups by chemistry cluster, and the two cross-cut each other. "
              "This inventory assumes the thesis adopts Paper A's definition. Keeping the parent-system definition would turn every ladder, ablation, "
              "external-validation and direct-versus-derived row into a new run.",
              "2. Dataset. The thesis dataset is a different snapshot and pipeline version from the Paper A snapshot, so every count and every R2 changes. "
              "No thesis number is carried over.",
              "3. Screening. The Materials Project screening and the oxide literature check are not in Paper A; they are the largest new-analysis items "
              "(NA11 and NA12). The supervisor's critical note on space groups means the current candidate list should not be reused.",
              "4. Sections to cut if time is short: the feature-selection experiment (Table 3), the JARVIS test and the Materials Project structural-feature "
              "claim have no code or result in the repository (see DROP_CANDIDATE rows).", ""]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT_CSV} ({len(allrows)} rows: {sum(c_claim.values())} claims, {len(issues)} issue notes)")
    print("claims:", dict(c_claim))
    print("issues:", dict(c_issue))
    if missing_in_summary:
        print("body notes not in summary:", missing_in_summary)


if __name__ == "__main__":
    main()
