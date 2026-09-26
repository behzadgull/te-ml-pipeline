"""
Stoichiometric-template family labeling for Paper B.

Assigns family labels (e.g. ABX half-Heusler, AB3 skutterudite) from
composition alone via template matching, since Starrydata2 carries no
structure data. Unmatched compositions are routed to an explicit
"unassignable" bucket rather than dropped, with its size reported.
Restricts the downstream family study to families above the a priori
sample-size threshold and reports what fraction of the dataset this
template-matched subset covers.

Labels are assigned to the HOST formula of the snapfix chemistry_cluster_id
(methodology doc, section 7 (d)), so a family is constant within a chemistry
cluster. The rules live in paper_b/config/families.yaml (v2.2; v1, v2 and v2.1 are
retained as families_v1.yaml, families_v2.yaml and families_v2_1.yaml); the a priori
threshold in paper_b/config/paper_b.yaml.
Run from the repository root:

    python -m paper_b.src.family_labels --print-rules
    python -m paper_b.src.family_labels --run --prev-run <v2 run> --gate <rules> <run> ...

No model is trained here. Shared imports are limited to those declared in
paper_b/SHARED_DEPENDENCIES.md.
"""

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import pandas as pd
import yaml

from src.canonicalization import parse_formula

MANIFEST_PATH = Path("paper_b/SHARED_DEPENDENCIES.md")
RULES_PATH = Path("paper_b/config/families.yaml")
PREV_RULES_PATH = Path("paper_b/config/families_v2_1.yaml")
THRESHOLD_PATH = Path("paper_b/config/paper_b.yaml")
REPORTS_DIR = Path("paper_b/reports/family_labels")
GROUP_COL = "chemistry_cluster_id"
TARGETS = ("S", "sigma", "kappa", "zT")
AUDIT_HOSTS_PER_FAMILY = 15
AUDIT_UNASSIGNABLE_HOSTS = 30
TOP_UNASSIGNABLE = 40
SEED = 0
_FLOAT_SLACK = 1e-9
SITE_TYPES = ("site_ratio", "ratio_range", "valence_balance")
OXIDE_TYPES = ("oxide", "element_fraction")  # element_fraction is the v1 spelling
RULE_TYPES = SITE_TYPES + ("element_set_only",) + OXIDE_TYPES


def templates(rule):
    """Return the rule's templates: its `variants` merged over the shared fields, or the rule itself."""
    if "variants" not in rule:
        return [rule]
    base = {key: value for key, value in rule.items() if key != "variants"}
    return [{**base, **variant} for variant in rule["variants"]]


def load_spec(path=RULES_PATH):
    """
    Load and validate a families yaml; return the spec dict (`rules`,
    `unassignable_label` and any notes).

    Validation: names are unique; rule types are known; element sets are
    disjoint within a template; site_ratio needs two or more required sites;
    ratio_range exactly two sites; valence_balance needs signed valences of
    both signs; `majority` elements lie on their site; every oxide-type rule
    comes after every other rule.
    """
    spec = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    rules = spec["rules"]
    names = [rule["name"] for rule in rules]
    if len(set(names)) != len(names):
        raise ValueError("duplicate rule names")
    seen_oxide = False
    for rule in rules:
        for template in templates(rule):
            kind = template["type"]
            if kind not in RULE_TYPES:
                raise ValueError(f"{rule['name']}: unknown rule type {kind!r}")
            if kind in OXIDE_TYPES:
                seen_oxide = True
                continue
            if seen_oxide:
                raise ValueError(f"{rule['name']}: non-oxide rule after an oxide rule")
            if kind == "element_set_only":
                continue
            elements = [el for site in template["sites"] for el in site["elements"]]
            if len(elements) != len(set(elements)):
                raise ValueError(f"{rule['name']}: an element appears on two sites")
            for site in template["sites"]:
                if not set(site.get("majority", [])) <= set(site["elements"]):
                    raise ValueError(f"{rule['name']}: majority element not on site {site['site']}")
            required = [s for s in template["sites"] if not s.get("optional")]
            if kind == "site_ratio" and len(required) < 2:
                raise ValueError(f"{rule['name']}: needs at least two required sites")
            if kind == "ratio_range" and len(template["sites"]) != 2:
                raise ValueError(f"{rule['name']}: ratio_range needs exactly two sites")
            if kind == "valence_balance":
                signs = {site["valence"] > 0 for site in template["sites"]}
                if signs != {True, False}:
                    raise ValueError(f"{rule['name']}: valence_balance needs cations and anions")
    return spec


def _site_lines(template):
    """Plain-language description of a template's sites."""
    parts = []
    for site in template["sites"]:
        els = "/".join(site["elements"])
        extra = ""
        if site.get("majority"):
            extra = f", majority element must be {'/'.join(site['majority'])}"
        if site.get("optional") and site.get("max_share"):
            share = site["max_share"]
            parts.append(
                f"optionally {els} on site {site['site']} (pooled with site {site['pooled_into']} as cations; "
                f"strictly below {share['below']} of the atoms on sites {'+'.join(share['of'])})"
            )
        elif site.get("optional"):
            parts.append(f"optionally {els} on site {site['site']} (at most {site['max_atom_fraction']:.0%} of all atoms)")
        elif "valence" in site:
            parts.append(f"{els} on site {site['site']} (valence {site['valence']:+d})")
        elif "count" in site:
            parts.append(f"{els} on site {site['site']} (count {site['count']}{extra})")
        else:
            parts.append(f"{els} on site {site['site']}")
    return "; ".join(parts)


def _describe_template(template):
    """One template in plain language."""
    kind = template["type"]
    if kind == "site_ratio":
        return (
            f"every atom must be one of: {_site_lines(template)}. Divide each required site's total "
            f"amount by its count; the largest and smallest results may differ by at most "
            f"{template['tolerance']:.0%} of their mean"
        )
    if kind == "ratio_range":
        return (
            f"every atom must be one of: {_site_lines(template)}. The amount of the second site per "
            f"amount of the first must lie between {template['min_ratio']} and {effective_max_ratio(template):.3f}"
            + (
                f" ({template['max_ratio']} / (1 - {template['metal_site_tolerance']}): the first site may be "
                f"up to {template['metal_site_tolerance']:.0%} deficient)"
                if template.get("metal_site_tolerance")
                else ""
            )
        )
    if kind == "valence_balance":
        share = template.get("min_share")
        return (
            f"every atom must be one of: {_site_lines(template)}, all sites non-empty. Total cation charge "
            f"and total anion charge may differ by at most {template['tolerance']:.0%} of their mean"
            + (
                f". Site {share['site']} must also hold at least {share['at_least']} of the atoms on "
                f"sites {'+'.join(share['of'])}"
                if share
                else ""
            )
        )
    if kind == "element_set_only":
        return f"every atom must be one of {'/'.join(template['elements'])}"
    text = f"oxygen makes up at least {template.get('min_o_fraction', template.get('min_atom_fraction')):.0%} of the atoms"
    if template.get("requires"):
        text += f"; {'/'.join(template['requires'])} must be present"
    if template.get("with_any"):
        text += f"; at least one of {'/'.join(template['with_any'])} must be present"
    if template.get("majority_cation"):
        text += f"; {template['majority_cation']} must be the single largest non-O element"
    if template.get("amount_exceeds"):
        text += f"; the amount of {template['amount_exceeds']['element']} must exceed the amount of {template['amount_exceeds']['over']}"
    return text + " (checked after every non-oxide rule)"


def describe_rule(rule):
    """Return one rule in plain language (what it accepts and why)."""
    variants = templates(rule)
    if len(variants) == 1:
        body = _describe_template(variants[0])
    else:
        body = "either " + "; or ".join(f"({_describe_template(v)})" for v in variants)
    return f"[{rule['name']}] {rule['description']}. {body[0].upper() + body[1:]}."


def _split_amounts(amounts, template):
    """Return (total atoms, per-site totals, fraction of atoms on no site) for a site-based template."""
    total = sum(amounts.values())
    owner = {el for site in template["sites"] for el in site["elements"]}
    site_amounts = {
        site["site"]: sum(amounts.get(el, 0.0) for el in site["elements"]) for site in template["sites"]
    }
    outside = {el for el in amounts if el not in owner}
    return total, site_amounts, outside


def _majority_failure(amounts, site):
    """Return why the site's largest element is not an allowed majority element, or None if it is."""
    present = sorted(((amounts.get(el, 0.0), el) for el in site["elements"] if amounts.get(el, 0.0) > 0), reverse=True)
    if not present:
        return f"site {site['site']} is empty"
    if len(present) > 1 and present[0][0] - present[1][0] <= _FLOAT_SLACK:
        return f"no single majority element on site {site['site']}"
    if present[0][1] not in site["majority"]:
        return f"majority element on site {site['site']} is {present[0][1]}, not {'/'.join(site['majority'])}"
    return None


def effective_max_ratio(template):
    """Upper limit of a ratio_range template: max_ratio, divided by (1 - metal_site_tolerance) if given."""
    return template["max_ratio"] / (1 - template.get("metal_site_tolerance", 0.0))


def _match_template(amounts, template):
    """
    Test one host (dict element -> amount) against one template.

    Returns (matched, detail); detail has `outside_fraction` (share of atoms on
    no site, None for oxide-type templates), `spread` (how far the ratio or
    charge balance is from the target; inf if not computable) and `why_not`.
    """
    kind = template["type"]
    total = sum(amounts.values())
    if kind in OXIDE_TYPES:
        o_fraction = amounts.get("O", 0.0) / total
        minimum = template.get("min_o_fraction", template.get("min_atom_fraction"))
        why = None
        if o_fraction + _FLOAT_SLACK < minimum:
            why = f"O fraction {o_fraction:.3f} < {minimum}"
        elif any(amounts.get(el, 0.0) <= 0 for el in template.get("requires", [])):
            why = f"missing one of {'/'.join(template['requires'])}"
        elif template.get("with_any") and not any(amounts.get(el, 0.0) > 0 for el in template["with_any"]):
            why = f"none of {'/'.join(template['with_any'])} present"
        elif template.get("majority_cation"):
            cations = sorted(((amt, el) for el, amt in amounts.items() if el != "O"), reverse=True)
            if not cations or cations[0][1] != template["majority_cation"] or (
                len(cations) > 1 and cations[0][0] - cations[1][0] <= _FLOAT_SLACK
            ):
                why = f"{template['majority_cation']} is not the single largest cation"
        if why is None and template.get("amount_exceeds"):
            element, over = template["amount_exceeds"]["element"], template["amount_exceeds"]["over"]
            if amounts.get(element, 0.0) - amounts.get(over, 0.0) <= _FLOAT_SLACK:
                why = f"{element} does not outweigh {over}"
        return why is None, {"outside_fraction": None, "spread": float("inf"), "why_not": why}
    if kind == "element_set_only":
        outside = {el: amt for el, amt in amounts.items() if el not in template["elements"]}
        fraction = sum(outside.values()) / total
        why = "elements outside rule: " + ",".join(sorted(outside)) if outside else None
        return not outside, {"outside_fraction": fraction, "spread": 0.0, "why_not": why}
    total, site_amounts, outside = _split_amounts(amounts, template)
    outside_fraction = sum(amounts[el] for el in outside) / total
    detail = {"outside_fraction": outside_fraction, "site_amounts": site_amounts, "spread": float("inf")}
    if outside:
        detail["why_not"] = "elements outside rule: " + ",".join(sorted(outside))
        return False, detail
    if kind == "site_ratio":
        for site in template["sites"]:
            limit = site.get("max_atom_fraction")
            if site.get("optional") and limit is not None and site_amounts[site["site"]] / total > limit + _FLOAT_SLACK:
                detail["why_not"] = f"{site['site']} above {site['max_atom_fraction']:.0%} of atoms"
                return False, detail
            share_rule = site.get("max_share")
            if site.get("optional") and share_rule:
                pool = sum(site_amounts[name] for name in share_rule["of"])
                share = site_amounts[site["site"]] / pool if pool else 0.0
                if share + _FLOAT_SLACK >= Fraction(share_rule["below"]):
                    detail["why_not"] = (
                        f"site {site['site']} is {share:.3f} of sites {'+'.join(share_rule['of'])}, "
                        f"not below {share_rule['below']}"
                    )
                    return False, detail
    required = [site for site in template["sites"] if not site.get("optional")]
    if any(site_amounts[site["site"]] <= 0 for site in required):
        detail["why_not"] = "a required site is empty"
        return False, detail
    if kind == "site_ratio":
        pooled = {name: 0.0 for name in site_amounts}
        for site in template["sites"]:
            if site.get("pooled_into"):
                pooled[site["pooled_into"]] += site_amounts[site["site"]]
        normalised = [(site_amounts[site["site"]] + pooled[site["site"]]) / site["count"] for site in required]
        mean = sum(normalised) / len(normalised)
        spread = (max(normalised) - min(normalised)) / mean
        limit = template["tolerance"]
        message = f"ratio spread {spread:.3f} > {limit}"
    elif kind == "valence_balance":
        cation = sum(site_amounts[s["site"]] * s["valence"] for s in template["sites"] if s["valence"] > 0)
        anion = -sum(site_amounts[s["site"]] * s["valence"] for s in template["sites"] if s["valence"] < 0)
        spread = abs(cation - anion) / ((cation + anion) / 2)
        limit = template["tolerance"]
        message = f"charge imbalance {spread:.3f} > {limit}"
    else:  # ratio_range
        first, second = template["sites"]
        ratio = site_amounts[second["site"]] / site_amounts[first["site"]]
        low, high = template["min_ratio"], effective_max_ratio(template)
        spread = 0.0 if low <= ratio <= high else (low - ratio if ratio < low else ratio - high) / ratio
        limit = 0.0
        message = f"{second['site']}/{first['site']} = {ratio:.3f} outside [{low}, {high:.3f}]"
    detail["spread"] = spread
    if spread > limit + _FLOAT_SLACK:
        detail["why_not"] = message
        return False, detail
    if kind == "valence_balance" and template.get("min_share"):
        share_rule = template["min_share"]
        pool = sum(site_amounts[name] for name in share_rule["of"])
        share = site_amounts[share_rule["site"]] / pool
        if share + _FLOAT_SLACK < Fraction(share_rule["at_least"]):
            detail["why_not"] = (
                f"site {share_rule['site']} is {share:.3f} of sites {'+'.join(share_rule['of'])}, "
                f"below {share_rule['at_least']}"
            )
            return False, detail
    for site in template["sites"]:
        if site.get("majority"):
            failure = _majority_failure(amounts, site)
            if failure:
                detail["why_not"] = failure
                return False, detail
    detail["why_not"] = None
    return True, detail


def match_rule(amounts, rule):
    """
    Test one host against one rule (any of its variants may match).

    Returns (matched, detail); when nothing matches, detail is the closest
    template's (fewest atoms outside its element sets, then smallest spread).
    """
    best = None
    for template in templates(rule):
        matched, detail = _match_template(amounts, template)
        if matched:
            return True, detail
        key = (float("inf") if detail["outside_fraction"] is None else detail["outside_fraction"], detail["spread"])
        if best is None or key < best[0]:
            best = (key, detail)
    return False, best[1]


def label_host(host, rules, unassignable_label):
    """
    Label one chemistry_cluster_id host formula.

    Returns a dict: family (first matching rule's name, else the
    unassignable label), all_matches (every rule that matches, in rule
    order, for the overlap audit), and, for unassignable hosts, the
    nearest non-oxide rule (fewest atoms outside its element sets, then
    smallest spread) and why it failed. Oxide-type rules are left out of the
    nearest-rule search: their "distance" is not comparable.
    """
    comp, error = parse_formula(host)
    if comp is None:
        raise ValueError(f"host {host!r} does not parse: {error}")
    amounts = comp.get_el_amt_dict()
    matches, near = [], []
    for rule in rules:
        matched, detail = match_rule(amounts, rule)
        if matched:
            matches.append(rule["name"])
        elif detail["outside_fraction"] is not None:
            near.append((detail["outside_fraction"], detail["spread"], rule["name"], detail["why_not"]))
    result = {"family": matches[0] if matches else unassignable_label, "all_matches": matches}
    if not matches:
        _, _, nearest_name, why = min(near, key=lambda item: (item[0], item[1]))
        result["nearest_rule"], result["why_not"] = nearest_name, why
    return result


def run_self_checks(spec):
    """Assert the spec's `self_checks` hosts get the labels they are defined to get (no-op if none)."""
    for host, expected in spec.get("self_checks", {}).items():
        got = label_host(host, spec["rules"], spec["unassignable_label"])["family"]
        if got != expected:
            raise AssertionError(f"self-check: {host} labelled {got!r}, expected {expected!r}")


def reason_key_for(prev_label):
    """Name of the per-rule note that explains a change from `prev_label` (v2.1 -> change_from_v2_1)."""
    return "change_from_" + prev_label.replace(".", "_")


def rules_diff(prev_spec, spec, reason_key="change_from_v2"):
    """
    Compare two specs rule by rule. Returns a DataFrame with one row per rule
    that is new, renamed, removed or changed (rule, status, changes, reason).
    Reasons come from `spec`: a rule's own `reason_key` note (default
    `change_from_v2`; the v1 -> v2 note key is `change_from_v1`), the shared
    `tolerance_change_reason` when the tolerance changed, `removed_from_v1` for
    removed rules. A rule with `renamed_from` is compared against that
    previous rule and reported as renamed.
    """
    ignore = {k for rule in spec["rules"] + prev_spec["rules"] for k in rule if k.startswith("change_from_")} | {"renamed_from"}
    prev = {rule["name"]: {k: v for k, v in rule.items() if k not in ignore} for rule in prev_spec["rules"]}
    compared_prev, rows = set(), []
    for rule in spec["rules"]:
        name = rule["name"]
        prev_name = name if name in prev else rule.get("renamed_from", name)
        own = {k: v for k, v in rule.items() if k not in ignore}
        reason = (rule.get(reason_key) or "").strip()
        if prev_name not in prev:
            rows.append({"rule": name, "status": "new", "changes": "new rule", "reason": reason})
            continue
        compared_prev.add(prev_name)
        skip = {"name"} if prev_name != name else set()
        changed = sorted(k for k in (set(own) | set(prev[prev_name])) - skip if own.get(k) != prev[prev_name].get(k))
        if not changed and prev_name == name:
            continue
        numeric = ("tolerance", "min_ratio", "max_ratio", "metal_site_tolerance", "min_share", "amount_exceeds")
        parts = [f"{k}: {prev[prev_name].get(k)} -> {own.get(k)}" for k in changed if k in numeric]
        parts += [f"{k} changed" for k in changed if k not in numeric]
        if prev_name != name:
            parts.insert(0, f"renamed from {prev_name}")
        if "tolerance" in changed and spec.get("tolerance_change_reason"):
            reason = (reason + " " + spec["tolerance_change_reason"].strip()).strip()
        status = "renamed" if prev_name != name else "changed"
        rows.append({"rule": name, "status": status, "changes": "; ".join(parts) or "position or wording only", "reason": reason})
    current = {rule["name"] for rule in spec["rules"]}
    for name in prev:
        if name not in compared_prev and name not in current:
            rows.append({"rule": name, "status": "removed", "changes": "rule removed",
                         "reason": spec.get("removed_from_v1", {}).get(name, "").strip()})
    return pd.DataFrame(rows)


def expected_dataset_identity(manifest_path=MANIFEST_PATH):
    """Read (path, sha256, bytes) of the snapfix CSV from the manifest's data table."""
    for line in Path(manifest_path).read_text(encoding="utf-8").splitlines():
        cells = [c.strip().strip("`") for c in line.strip().strip("|").split("|")]
        if len(cells) == 4 and cells[0] == "snapfix featurized CSV":
            return Path(cells[1]), cells[2], int(cells[3].replace(",", ""))
    raise ValueError(f"no snapfix data row in {manifest_path}")


def sha256_file(path, chunk=1 << 24):
    """SHA256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def load_cluster_table():
    """
    Load the snapfix CSV by explicit path after a SHA256 and size check, and
    return (cluster table, dataset identity). The table has one row per
    chemistry_cluster_id with `rows` (all rows) and, per target, the number
    of rows with a non-null value. Does not use src.nested_cv.load_target_data.
    """
    path, expected_sha, expected_bytes = expected_dataset_identity()
    size = path.stat().st_size
    if size != expected_bytes:
        raise ValueError(f"{path}: {size} bytes, manifest says {expected_bytes}")
    sha = sha256_file(path)
    if sha != expected_sha:
        raise ValueError(f"{path}: SHA256 {sha} differs from the manifest {expected_sha}")
    df = pd.read_csv(path, usecols=[GROUP_COL, *TARGETS])
    if df[GROUP_COL].isna().any():
        raise ValueError("null chemistry_cluster_id in the snapfix CSV")
    table = df.groupby(GROUP_COL).agg(
        rows=("S", "size"), **{target: (target, "count") for target in TARGETS}
    )
    identity = {"path": str(path), "sha256": sha, "bytes": size, "n_rows": int(len(df))}
    return table, identity


def label_all_hosts(table, rules, unassignable_label):
    """Label every host in the cluster table; return the table with label columns added."""
    labels = [label_host(host, rules, unassignable_label) for host in table.index]
    out = table.copy()
    out["family"] = [item["family"] for item in labels]
    out["n_rules_matched"] = [len(item["all_matches"]) for item in labels]
    out["all_matches"] = [";".join(item["all_matches"]) for item in labels]
    out["nearest_rule"] = [item.get("nearest_rule", "") for item in labels]
    out["why_not"] = [item.get("why_not") or "" for item in labels]
    return out


def family_summary(labelled, threshold, unassignable_label):
    """
    Per-family cluster and row counts (all clusters, and per target the
    clusters with at least one row plus the rows), and whether the family
    qualifies for each target under the a priori threshold: at least
    min_clusters clusters holding data for that target AND at least
    min_rows_per_target rows for it. The unassignable bucket is summarised
    but never marked as qualifying.
    """
    rows = []
    for family, group in labelled.groupby("family"):
        row = {"family": family, "clusters_all": len(group), "rows_all": int(group["rows"].sum())}
        for target in TARGETS:
            row[f"clusters_{target}"] = int((group[target] > 0).sum())
            row[f"rows_{target}"] = int(group[target].sum())
            row[f"qualifies_{target}"] = bool(
                family != unassignable_label
                and row[f"clusters_{target}"] >= threshold["min_clusters"]
                and row[f"rows_{target}"] >= threshold["min_rows_per_target"]
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("rows_all", ascending=False).reset_index(drop=True)


def check_reproduces(rules_path, run_dir):
    """
    Regression gate: this code, given the rules in `rules_path`, must reproduce
    the labels of the retained run in `run_dir` for every host in it. Returns
    that run's label Series (index host, value family); raises if any host differs.
    """
    spec = load_spec(rules_path)
    old = pd.read_csv(
        Path(run_dir) / "host_family_labels.csv", usecols=["host", "family"], keep_default_na=False
    ).set_index("host")["family"]
    redone = pd.Series({host: label_host(host, spec["rules"], spec["unassignable_label"])["family"] for host in old.index})
    differing = int((redone != old.loc[redone.index]).sum())
    if differing:
        raise AssertionError(f"{differing} hosts labelled differently by {rules_path} under the current code than in {run_dir}")
    return old


def check_expected_moves(prev_labels, labelled, expected):
    """
    Assert that every host whose label differs from the previous version's moved
    from one of expected["from"] to one of expected["to"] and contains at least
    one element from each group in expected["host_contains_each_of"]. Returns the
    number of hosts that moved; raises listing the offenders otherwise.
    """
    current = labelled["family"].reindex(prev_labels.index)
    moved = prev_labels.index[(prev_labels != current).to_numpy()]
    offenders = []
    for host in moved:
        elements = set(parse_formula(host)[0].get_el_amt_dict())
        ok = (
            prev_labels[host] in expected["from"]
            and current[host] in expected["to"]
            and all(elements & set(group) for group in expected["host_contains_each_of"])
        )
        if not ok:
            offenders.append(f"{host}: {prev_labels[host]} -> {current[host]}")
    if offenders:
        raise AssertionError(f"{len(offenders)} unexpected moves, e.g. " + "; ".join(offenders[:10]))
    return len(moved)


def transition_matrix(prev_labels, labelled, weight=None):
    """
    Cluster counts (or the summed `weight` column, e.g. rows) by previous family
    (rows of the result) and current family (columns), over the hosts common to both.
    """
    current = labelled["family"].reindex(prev_labels.index)
    if weight is None:
        return pd.crosstab(prev_labels.rename("prev_family"), current.rename("family"))
    return pd.crosstab(
        prev_labels.rename("prev_family"), current.rename("family"),
        values=labelled[weight].reindex(prev_labels.index), aggfunc="sum",
    ).fillna(0).astype(int)


def audit_sample(labelled, rules, unassignable_label, seed=SEED):
    """
    Random hosts for manual review: AUDIT_HOSTS_PER_FAMILY per family (all of
    them if a family is smaller) and AUDIT_UNASSIGNABLE_HOSTS unassignable
    hosts, each with the rule that matched it (or, for unassignable hosts,
    the nearest rule and why it failed).
    """
    rule_by_name = {rule["name"]: rule for rule in rules}
    picks = []
    for family, group in labelled.groupby("family"):
        n = AUDIT_UNASSIGNABLE_HOSTS if family == unassignable_label else AUDIT_HOSTS_PER_FAMILY
        picks.append(group.sample(n=min(n, len(group)), random_state=seed))
    sample = pd.concat(picks)
    out = sample.reset_index()[[GROUP_COL, "family", "all_matches", "nearest_rule", "why_not", "rows", *TARGETS]]
    out = out.rename(columns={GROUP_COL: "host", "family": "assigned_family", "all_matches": "rules_matched"})
    out.insert(2, "matched_rule", out["assigned_family"].where(out["assigned_family"] != unassignable_label, "(none)"))
    out["rule_description"] = out["matched_rule"].map(
        lambda name: rule_by_name[name]["description"] if name in rule_by_name else ""
    )
    return out.sort_values(["assigned_family", "host"]).reset_index(drop=True)


def git_state():
    """Return (HEAD sha, tree_clean, dirty file list) for the run_config."""
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    porcelain = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout
    dirty = [line[3:] for line in porcelain.splitlines()]
    return head, not dirty, dirty


def print_rules(spec, prev_spec, prev_label="v2.1", label="v2.2"):
    """Print every rule in plain language, then the previous -> current diff table (new, renamed, changed and removed rules)."""
    rules = spec["rules"]
    print(f"{len(rules)} rules, first match wins, unmatched -> '{spec['unassignable_label']}':\n")
    for number, rule in enumerate(rules, 1):
        print(f"{number}. {describe_rule(rule)}\n")
    if prev_spec is None:
        return
    diff = rules_diff(prev_spec, spec, reason_key=reason_key_for(prev_label))
    unchanged = [r["name"] for r in rules if r["name"] not in set(diff["rule"])]
    print(f"{prev_label} -> {label}: {len(diff)} rules new, renamed, changed or removed; unchanged: {', '.join(unchanged) or 'none'}\n")
    for _, row in diff.iterrows():
        print(f"- {row['rule']} [{row['status']}] {row['changes']}\n    reason: {row['reason']}")
    print()


def print_report(labelled, summary, unassignable, unassignable_label, threshold, transitions, prev_label="v2.1", label="v2.2"):
    """Print the per-family table, unassignable share, qualifying families, previous -> current transitions and top unassignable hosts."""
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 200)
    print(f"Threshold: >= {threshold['min_clusters']} clusters holding the target AND >= "
          f"{threshold['min_rows_per_target']} rows for it.\n")
    print("Per family (clusters_X = clusters with at least one row for X; rows_X = rows with X):")
    columns = ["family", "clusters_all", *[f"clusters_{t}" for t in TARGETS], *[f"rows_{t}" for t in TARGETS]]
    print(summary[columns].to_string(index=False), "\n")
    totals = labelled[["rows", *TARGETS]].sum()
    n_hosts = len(labelled)
    print(f"Unassignable: {len(unassignable)} of {n_hosts} clusters ({len(unassignable) / n_hosts:.1%}); rows:")
    for column in ("rows", *TARGETS):
        share = unassignable[column].sum() / totals[column]
        print(f"  {column:>6}: {int(unassignable[column].sum()):>8,} of {int(totals[column]):>8,} ({share:.1%})")
    print("\nQualifying families per target:")
    for target in TARGETS:
        names = summary.loc[summary[f"qualifies_{target}"], "family"].tolist()
        print(f"  {target:>5}: {len(names)}  {', '.join(names)}")
    print("\nNot qualifying (family: targets failed):")
    for _, row in summary[summary["family"] != unassignable_label].iterrows():
        failed = [t for t in TARGETS if not row[f"qualifies_{t}"]]
        if failed:
            print(f"  {row['family']}: {', '.join(failed)}")
    multi = labelled[labelled["n_rules_matched"] > 1]
    print(f"\nHosts matching more than one rule: {len(multi)}")
    if len(multi):
        print(multi["all_matches"].value_counts().to_string())
    if transitions is not None:
        print(f"\n{prev_label} -> {label} transition, clusters ({prev_label} family (total): {label} family n):")
        for prev_family, row in transitions.iterrows():
            moved = row[row > 0].sort_values(ascending=False)
            print(f"  {prev_family} ({int(row.sum())}): " + ", ".join(f"{fam} {n}" for fam, n in moved.items()))
    print(f"\n{TOP_UNASSIGNABLE} largest unassignable hosts by rows (nearest rule and why not):")
    top = unassignable.head(TOP_UNASSIGNABLE).reset_index().rename(columns={GROUP_COL: "host"})
    print(top[["host", "rows", *TARGETS, "nearest_rule", "why_not"]].to_string(index=False))


def main(argv=None):
    """CLI entry point: print the rules, or label the snapfix hosts and write the report files."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--rules", default=str(RULES_PATH), help="rules yaml to run (default: families.yaml, v2.2)")
    parser.add_argument("--label", default="v2.2", help="name of this rule version, used in file names")
    parser.add_argument("--prev-rules", default=str(PREV_RULES_PATH), help="previous version's rules yaml, for the diff table")
    parser.add_argument("--prev-label", default="v2.1", help="name of the previous rule version")
    parser.add_argument("--prev-run", default=None, help="previous version's run folder, for the transition matrix")
    parser.add_argument("--gate", nargs=2, action="append", default=[], metavar=("RULES", "RUN"),
                        help="regression gate: these rules under the current code must reproduce that run's labels (repeatable)")
    parser.add_argument("--audit-seed", type=int, default=SEED, help="random seed for the audit sample (default 0)")
    parser.add_argument("--print-rules", action="store_true", help="print every rule in plain language and the diff, then exit")
    parser.add_argument("--run", action="store_true", help="label every host and write paper_b/reports/family_labels/<UTC>/")
    args = parser.parse_args(argv)
    spec = load_spec(args.rules)
    prev_spec = load_spec(args.prev_rules) if Path(args.prev_rules).exists() and Path(args.prev_rules) != Path(args.rules) else None
    print_rules(spec, prev_spec, args.prev_label, args.label)
    if not args.run:
        return
    rules, unassignable_label = spec["rules"], spec["unassignable_label"]
    run_self_checks(spec)
    gates = []
    for gate_rules, gate_run in args.gate:
        check_reproduces(gate_rules, gate_run)
        gates.append({"rules": gate_rules, "rules_sha256": sha256_file(gate_rules), "run": gate_run, "passed": True})
        print(f"Regression gate passed: {gate_rules} reproduces {gate_run}")
    prev_labels = check_reproduces(args.prev_rules, args.prev_run) if args.prev_run else None
    threshold = yaml.safe_load(THRESHOLD_PATH.read_text(encoding="utf-8"))["paper_b"]["min_family_sample_size"]
    table, identity = load_cluster_table()
    labelled = label_all_hosts(table, rules, unassignable_label)
    if prev_labels is not None and set(prev_labels.index) != set(labelled.index):
        raise AssertionError("the previous run and this run cover different hosts")
    expected = spec.get("expected_moves")
    n_moved = None
    if expected and prev_labels is not None:
        n_moved = check_expected_moves(prev_labels, labelled, expected)
        print(f"Expected-moves assertion passed: {n_moved} hosts moved, all from {'/'.join(expected['from'])} "
              f"to {'/'.join(expected['to'])}, each containing an element from every group in host_contains_each_of.")
    summary = family_summary(labelled, threshold, unassignable_label)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out_dir = REPORTS_DIR / stamp
    out_dir.mkdir(parents=True)
    head, tree_clean, dirty = git_state()
    config = {
        "version_label": args.label,
        "dataset": identity,
        "git_head": head,
        "tree_clean": tree_clean,
        "dirty_files": dirty,
        "rules_file": str(args.rules),
        "rules_sha256": sha256_file(args.rules),
        "prev_label": args.prev_label if prev_spec else None,
        "prev_rules_file": str(args.prev_rules) if prev_spec else None,
        "prev_rules_sha256": sha256_file(args.prev_rules) if prev_spec else None,
        "prev_run": str(args.prev_run) if args.prev_run else None,
        "regression_gates": gates,
        "expected_moves": expected if n_moved is not None else None,
        "n_hosts_moved_from_prev": n_moved,
        "code_sha256": sha256_file(Path(__file__)),
        "threshold": threshold,
        "threshold_file": str(THRESHOLD_PATH),
        "audit_seed": args.audit_seed,
        "n_hosts": int(len(labelled)),
        "utc_stamp": stamp,
    }
    (out_dir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    labelled.reset_index().rename(columns={GROUP_COL: "host"}).to_csv(out_dir / "host_family_labels.csv", index=False)
    summary.to_csv(out_dir / "family_summary.csv", index=False)
    unassignable = labelled[labelled["family"] == unassignable_label].sort_values("rows", ascending=False)
    unassignable.head(TOP_UNASSIGNABLE).reset_index().rename(columns={GROUP_COL: "host"}).to_csv(
        out_dir / "unassignable_top40.csv", index=False
    )
    audit_sample(labelled, rules, unassignable_label, seed=args.audit_seed).to_csv(out_dir / "audit_sample.csv", index=False)
    (out_dir / "rules_plain_language.txt").write_text(
        "\n\n".join(f"{n}. {describe_rule(r)}" for n, r in enumerate(rules, 1)) + "\n", encoding="utf-8"
    )
    transitions = None
    pair = f"{args.prev_label}_to_{args.label}"
    if prev_spec:
        rules_diff(prev_spec, spec, reason_key=reason_key_for(args.prev_label)).to_csv(out_dir / f"rules_diff_{pair}.csv", index=False)
    if prev_labels is not None:
        transitions = transition_matrix(prev_labels, labelled)
        transitions.to_csv(out_dir / f"transition_{pair}_clusters.csv")
        transition_matrix(prev_labels, labelled, weight="rows").to_csv(out_dir / f"transition_{pair}_rows.csv")
    print(f"Wrote {out_dir}\n")
    print_report(labelled, summary, unassignable, unassignable_label, threshold, transitions, args.prev_label, args.label)


if __name__ == "__main__":
    main()
