"""
Fill every number-bearing part of paper_b/paper/paper.md from the committed Paper B artifacts, so that no
value in the manuscript is typed by hand.

Three mechanisms, all rewritten in place by this script:
  1. Inline values: `<!--v:key-->text<!--/v-->`. The text between the markers is replaced by the value
     computed here for `key` (see `inline_values`). Pandoc drops the markers from the docx.
  2. Table blocks: `<!-- BEGIN TABLE n -->` ... `<!-- END TABLE n -->` hold Table 1 (family rules, example
     hosts, clusters and rows) and Table 2 (conditions), caption included.
  3. Prose block: `<!-- BEGIN SECTION 3.1 -->` ... `<!-- END SECTION 3.1 -->` holds Section 3.1, written
     from the final labels and the super-family qualification.

Every value is derived from paper_b/scripts/paper_b_facts.py, which cross-checks the artifacts against
each other before returning anything.

Usage (from the repository root):
    python paper_b/scripts/make_tables_b.py           # rewrite paper.md
    python paper_b/scripts/make_tables_b.py --check   # exit 1 if paper.md is out of date or has an unknown key
"""

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from paper_b.scripts import paper_b_facts as pf  # noqa: E402

PAPER = REPO / "paper_b" / "paper" / "paper.md"
LINUX_SMOKE = REPO / "paper_b" / "results" / "smoke_test_linux"

WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}
EXAMPLE_HOSTS = 3


def n(x):
    """Integer with thousands separators."""
    return f"{int(x):,}"


def pct(num, den, places=1):
    """Percentage string of num/den."""
    return f"{100.0 * num / den:.{places}f}%"


def fam(key):
    """Plain family name, as pandoc markdown."""
    return pf.FAMILY_NAMES[key][0]


def join_and(items):
    """'a, b and c'."""
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def latest_linux_smoke():
    """(number of checks, all_passed, commit) of the newest committed Linux smoke test of the harness."""
    runs = sorted(p for p in LINUX_SMOKE.iterdir() if (p / "report_linux.json").exists() and (p / "README.md").exists())
    run = runs[-1]
    report = json.loads((run / "report_linux.json").read_text(encoding="utf-8"))
    checks = [k for k in report if re.fullmatch(r"check_\d+_.*", k)]
    commit = re.search(r"\*\*Commit\*\*: `([0-9a-f]{40})`", (run / "README.md").read_text(encoding="utf-8")).group(1)
    return len(checks), bool(report["all_passed"]), commit


def tiers(f):
    """Derived counts shared by the values and the prose."""
    summ = f["summary"].set_index("family")
    named = [k for k in pf.RULE_ORDER if k not in pf.NEVER_HELD_OUT]
    assert len(named) == f["n_named_families"] == 27
    qual = {t: [k for k in named if bool(summ.loc[k, f"qualifies_{t}"])] for t in pf.TARGETS}
    for t in pf.TARGETS:
        assert len(qual[t]) == f["qualifying"][t]
    pairs = sum(len(v) for v in qual.values())
    assert pairs == f["splits_cfg"]["n_family_pairs"]
    return summ, named, qual, pairs


def inline_values(f):
    """Dictionary key -> text for every inline `<!--v:key-->` marker in paper.md."""
    summ, named, qual, pairs = tiers(f)
    un, oo, asg = f["unassignable"], f["other_oxide"], f["assigned_to_named_family"]
    cfg = f["splits_cfg"]
    n_checks, passed, commit = latest_linux_smoke()
    assert passed
    audit_default = {v for k, v in f["audit_n"].items() if k not in ("unassignable", "mgagsb")}
    assert audit_default == {15}, audit_default
    assert f["audit_n"]["unassignable"] == 30
    assert f["labels_cfg"]["tree_clean"] is True
    assert f["min_clusters"] % f["n_folds"] == 0
    v = {
        "n_named": str(f["n_named_families"]),
        "n_rows": n(f["n_rows"]),
        "n_hosts": n(f["n_hosts"]),
        "un_row_share": pct(un["rows"], f["n_rows"]),
        "un_cluster_share": pct(un["clusters"], f["n_hosts"]),
        "assigned_row_share": pct(asg["rows"], f["n_rows"]),
        "min_clusters": str(f["min_clusters"]),
        "min_rows": n(f["min_rows"]),
        "min_rows_per_fold": n(f["min_rows"] // f["n_folds"]),
        "clusters_per_fold_word": WORDS[f["min_clusters"] // f["n_folds"]],
        "n_folds_word": WORDS[f["n_folds"]],
        "repeats_word": WORDS[f["repeats"]],
        "audit_n": str(15),
        "audit_n_unassignable": str(f["audit_n"]["unassignable"]),
        "n_family_pairs": str(pairs),
        "n_super_pairs": str(cfg["n_super_family_pairs"]),
        "n_standalone_pairs": str(cfg["n_standalone_pairs"]),
        "n_standalone_families": str(len(f["standalone"])),
        "n_checks_word": WORDS.get(n_checks, str(n_checks)),
        "linux_commit": commit[:7],
        "labels_version": f["labels_cfg"]["version_label"],
    }
    for t in pf.TARGETS:
        v[f"q_{t}"] = str(len(qual[t]))
    return v


def table1(f):
    """Table 1 (caption plus pipe table) as markdown."""
    summ, named, qual, pairs = tiers(f)
    audit = f["audit"]
    lines = [
        f"**Table 1.** Family rules and held-out units. Rules are listed in the order applied; the first match wins. "
        f"Examples are the {WORDS[EXAMPLE_HOSTS]} shortest host formulas in each family's random audit sample, with dopants "
        f"already removed (Section 2.2). Rows are rows with a value for the property; italic entries are below the "
        f"threshold of {n(f['min_clusters'])} chemistry clusters and {n(f['min_rows'])} rows for that property "
        f"(Section 2.4). The unassignable and other-oxide categories are never held out.",
        "",
        "| Family | Rule | Example hosts | Clusters | S rows | σ rows | κ rows | zT rows |",
        "|--------------|----------------------------------------|----------------------|--------|-------|-------|-------|-------|",
    ]
    tot_rows = {t: 0 for t in pf.TARGETS}
    tot_clusters = 0
    for key in list(pf.RULE_ORDER) + ["unassignable"]:
        r = summ.loc[key]
        if key == "unassignable":
            name, rule, ex = "Unassignable", "No rule matches", "none"
        else:
            name, rule = fam(key), pf.RULE_WORDS[key]
            sub = audit[audit["assigned_family"] == key]
            hosts = sorted(sub.assign(_l=sub["host"].str.len()).sort_values(["_l", "host"])["host"].head(EXAMPLE_HOSTS), key=lambda h: (len(h), h))
            ex = ", ".join(pf.format_host(h) for h in hosts)
        cells = []
        for t in pf.TARGETS:
            val = n(r[f"rows_{t}"])
            held_out = key not in pf.NEVER_HELD_OUT
            cells.append(f"*{val}*" if held_out and not bool(r[f"qualifies_{t}"]) else val)
            tot_rows[t] += int(r[f"rows_{t}"])
        tot_clusters += int(r["clusters_all"])
        lines.append(f"| {name} | {rule} | {ex} | {n(r['clusters_all'])} | " + " | ".join(cells) + " |")
    assert tot_clusters == f["n_hosts"]
    for t in pf.TARGETS:
        assert tot_rows[t] == f["target_rows"][t], (t, tot_rows[t])
    lines.append(f"| **All** | | | **{n(tot_clusters)}** | " + " | ".join(f"**{n(tot_rows[t])}**" for t in pf.TARGETS) + " |")
    return "\n".join(lines)


def table2(f):
    """Table 2 (caption plus pipe table) as markdown."""
    lines = [
        "**Table 2.** Training data under each condition, for one held-out family *F*, property and test fold. "
        "All conditions are scored on the same test fold of *F* (Section 2.5).",
        "",
        "| Condition | Training data | What it tests |",
        "|-----------------|--------------------------------------|--------------------------------------|",
        "| C0, pooled | All rows except the test fold | Within-distribution accuracy on *F*, with the rest of *F* in training |",
        "| C1, leave-one-family-out | All rows except every row of *F* | Accuracy on a family the model has never seen |",
        "| C2, size-matched random removal | C0 minus a random sample of rows from outside *F*, so that the training set has the size of C1's | Whether C1's loss is explained by having fewer training rows |",
        "| C3, structured removal | C0 minus one whole other family *G* (closest to *F* in row count, outside *F*'s super-family) | Whether losing any coherent region of chemistry costs as much as losing *F*'s relatives |",
        "| Specialist | The rest of *F* only (*F*'s training folds) | Whether a model trained on *F* alone beats the pooled model on the same test folds |",
    ]
    return "\n".join(lines)


def section31(f):
    """Section 3.1 body, written from the final labels and the super-family qualification."""
    summ, named, qual, pairs = tiers(f)
    un, oo, asg = f["unassignable"], f["other_oxide"], f["assigned_to_named_family"]
    N, H = f["n_rows"], f["n_hosts"]
    assert asg["clusters"] + oo["clusters"] + un["clusters"] == H and asg["rows"] + oo["rows"] + un["rows"] == N
    share = un["row_share_by_target"]
    lo = min(share, key=share.get)
    hi = max(share, key=share.get)
    sym = pf.TARGET_SYMBOL

    by_rows = sorted(named, key=lambda k: -int(summ.loc[k, "rows_all"]))
    top3 = by_rows[:3]
    top3_rows = sum(int(summ.loc[k, "rows_all"]) for k in top3)
    big = by_rows[0]
    small = sorted(named, key=lambda k: (int(summ.loc[k, "rows_all"]), k))[0]

    # failure reasons per family-property pair
    reasons = {"rows only": 0, "clusters only": 0, "both": 0}
    for t in pf.TARGETS:
        fl = f["failing"][t]
        reasons["rows only"] += len(fl["rows_only"])
        reasons["clusters only"] += len(fl["clusters_only"])
        reasons["both"] += len(fl["both"])
        assert set(fl["rows_only"]) | set(fl["clusters_only"]) | set(fl["both"]) == set(named) - set(qual[t])
    n_fail = sum(reasons.values())
    assert n_fail == 4 * len(named) - pairs
    all4 = [k for k in named if all(k in qual[t] for t in pf.TARGETS)]
    none4 = [k for k in named if all(k not in qual[t] for t in pf.TARGETS)]

    cover = {}
    for t in pf.TARGETS:
        cover[t] = sum(int(summ.loc[k, f"rows_{t}"]) for k in qual[t])
    cov_txt = join_and(f"{pct(cover[t], f['target_rows'][t])} of the {sym[t]} rows" for t in pf.TARGETS)

    sup = f["supers"]
    assert all(sup[s]["qualifies"][t] for s in sup for t in pf.TARGETS)
    union_rows = [sup[s]["rows"][t] for s in sup for t in pf.TARGETS]
    below = []
    for s in sup:
        mem = {}
        for t in pf.TARGETS:
            for m in sup[s]["below_alone"][t]:
                mem.setdefault(m, []).append(sym[t])
        for m, ts in mem.items():
            below.append(f"{fam(m)} for {join_and(ts) if len(ts) < 4 else 'all four properties'}")
    n_members = sum(len(v) for v in f["super_members"].values())
    assert len(f["standalone"]) == len(named) - n_members
    sq = f["standalone_qualifying"]

    hosts = f["hosts"]
    unh = hosts[hosts["family"] == "unassignable"]
    assert len(unh) == un["clusters"]
    reason = unh["why_not"].fillna("").map(lambda r: re.split(r"[:0-9]", r)[0].strip())
    rc = reason.value_counts()
    why = (
        f"For each unassignable host, the nearest rule fails because an element lies outside it for "
        f"{pct(rc['elements outside rule'], len(unh))} of hosts, because a site ratio is outside tolerance for "
        f"{pct(rc['ratio spread'], len(unh))} and because a required site is empty for "
        f"{pct(rc['a required site is empty'], len(unh))}; the rest fail on other conditions of that rule."
    )

    p = []
    p.append(
        f"The labelled snapshot has {n(N)} rows in {n(H)} chemistry clusters. The {len(named)} family rules assign "
        f"{n(asg['clusters'])} clusters ({pct(asg['clusters'], H)}) and {n(asg['rows'])} rows ({pct(asg['rows'], N)}) to a "
        f"named family, and the other-oxide rule a further {n(oo['clusters'])} clusters ({pct(oo['clusters'], H)}) and "
        f"{n(oo['rows'])} rows ({pct(oo['rows'], N)}). The remaining {n(un['clusters'])} clusters ({pct(un['clusters'], H)}) and "
        f"{n(un['rows'])} rows ({pct(un['rows'], N)}) match no rule and are labelled unassignable (Table 1, Figure 2b). "
        f"The unassignable share of rows is similar across properties, from {pct(share[lo], 1)} for {sym[lo]} to "
        f"{pct(share[hi], 1)} for {sym[hi]}. "
        f"{why} Unassignable hosts stay in every training pool but are never a held-out family."
    )
    p.append(
        f"Family sizes are very unequal (Figure 2a). {fam(big)} is the largest, with {n(summ.loc[big, 'clusters_all'])} clusters "
        f"and {n(summ.loc[big, 'rows_all'])} rows; the three largest families ({join_and(fam(k) for k in top3)}) hold "
        f"{pct(top3_rows, asg['rows'])} of the rows assigned to a named family. At the other end, {fam(small)} has "
        f"{n(summ.loc[small, 'clusters_all'])} cluster{'s' if int(summ.loc[small, 'clusters_all']) != 1 else ''} and "
        f"{n(summ.loc[small, 'rows_all'])} rows."
    )
    p.append(
        f"A family qualifies for a property if it has at least {n(f['min_clusters'])} chemistry clusters and "
        f"{n(f['min_rows'])} rows with a value for that property (Section 2.4). Of the {len(named)} named families, "
        f"{join_and(f'{len(qual[t])} qualify for {sym[t]}' for t in pf.TARGETS)}, giving {pairs} family-property units "
        f"(Figure 3). Of these families, {len(all4)} qualify for all four properties and {len(none4)} for none "
        f"({join_and(fam(k) for k in none4)}). Of the {n_fail} family-property pairs that fall below the threshold, "
        f"{reasons['rows only']} fail on rows only, {reasons['clusters only']} on clusters only and {reasons['both']} on both. "
        f"Together, the qualifying families hold {cov_txt}, so held-out results describe these slices of "
        f"each property's data, not all of it."
    )
    p.append(
        f"The three super-families (Section 2.8) qualify for all four properties as unions, giving {f['splits_cfg']['n_super_family_pairs']} "
        f"super-family units with between {n(min(union_rows))} and {n(max(union_rows))} rows each. Several members that fall below the "
        f"threshold alone are carried by these unions: {join_and(below)}. The other {len(f['standalone'])} families belong to no "
        f"super-family and are standalone units, with {join_and(f'{sq[t]} qualifying for {sym[t]}' for t in pf.TARGETS)} "
        f"({f['splits_cfg']['n_standalone_pairs']} units in all); they are rerun for the structured-removal control only."
    )
    return "\n\n".join(p)


BLOCKS = {
    "TABLE 1": table1,
    "TABLE 2": table2,
    "SECTION 3.1": section31,
}


def render(text, f):
    """Return `text` with all inline values and blocks recomputed. Raises KeyError on an unknown key."""
    vals = inline_values(f)

    def sub_inline(m):
        key = m.group(1)
        if key not in vals:
            raise KeyError(f"unknown inline value key {key!r}")
        return f"<!--v:{key}-->{vals[key]}<!--/v-->"

    text = re.sub(r"<!--v:(\w+)-->(.*?)<!--/v-->", sub_inline, text)
    for name, fn in BLOCKS.items():
        pat = re.compile(rf"(<!-- BEGIN {re.escape(name)} -->\n)(?:.*?\n)?(<!-- END {re.escape(name)} -->)", re.S)
        if not pat.search(text):
            raise KeyError(f"block {name!r} not found in paper.md")
        body = fn(f) + "\n"
        text = pat.sub(lambda m, body=body: m.group(1) + body + m.group(2), text)
    return text


def main():
    """Entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="exit 1 if paper.md is out of date")
    ap.add_argument("--paper", default=str(PAPER))
    args = ap.parse_args()
    path = Path(args.paper)
    old = path.read_bytes().decode("utf-8")
    f = pf.load_facts()
    new = render(old.replace("\r\n", "\n"), f)
    if args.check:
        if new != old.replace("\r\n", "\n"):
            print("paper.md is out of date: run python paper_b/scripts/make_tables_b.py")
            return 1
        print("paper.md values and blocks are current")
        return 0
    path.write_bytes(new.encode("utf-8"))
    print(f"wrote {path} ({len(re.findall('<!--v:', new))} inline values, {len(BLOCKS)} blocks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
