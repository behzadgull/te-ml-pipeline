# Per-GROUP breakdown of the already-saved teMatDb scoring predictions
# (results/20260910T123047_tematdb_external/tematdb_{a0,a,b}_predictions.npz,
# File B): R2/RMSE/MAE per GROUP per property/stratum, plus the sigma
# leave-one/two-group-out diagnostic in stratum b-and-a-CLEAN that CLAUDE.md's
# CAVEATS (a) cites. Read-only, no model touch. Produced
# results/20260910T123047_tematdb_external/per_group_breakdown.json.
import json
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

MIN_SAMPLES = 3
RESULTS_DIR = "results/20260910T123047_tematdb_external"

# --- load GROUP mapping ---
samples_df = pd.read_csv("data/external/tematdb/teMatDb_samples.csv")
sample_to_group = dict(zip(samples_df["sample_id"], samples_df["GROUP"]))

# --- load per-row predictions ---
def load_df(stratum):
    d = np.load(f"{RESULTS_DIR}/tematdb_{stratum}_predictions.npz", allow_pickle=True)
    df = pd.DataFrame({k: d[k] for k in d.keys()})
    df["GROUP"] = df["sample_id"].map(sample_to_group)
    return df

df_a0 = load_df("a0")
df_a = load_df("a")
df_b = load_df("b")

CLEAN_SAMPLE_IDS = [15, 91, 119, 141, 147, 153, 173, 201, 216, 228, 272, 302,
                    360, 361, 362, 364, 379, 392, 393, 396, 397, 406, 411, 418, 419]
df_b_and_a = df_b[df_b["sample_id"].isin(CLEAN_SAMPLE_IDS)].copy()

STRATA = {
    "a0": df_a0,
    "a": df_a,
    "b": df_b,
    "b_and_a_CLEAN": df_b_and_a,
}

# property -> (true_col, pred_col, space)
PROPERTIES = {
    "S": ("S_true", "S_pred", "linear"),
    "sigma_log10": ("sigma_log_true", "sigma_log_pred", "log10"),
    "kappa_log10": ("kappa_log_true", "kappa_log_pred", "log10"),
    "zT_direct_vs_declared": ("zt_declared_true", "zT_direct_pred", "linear"),
    "zT_direct_vs_tep": ("zt_tep_true", "zT_direct_pred", "linear"),
    "zT_derived_vs_declared": ("zt_declared_true", "zT_derived_pred", "linear"),
    "zT_derived_vs_tep": ("zt_tep_true", "zT_derived_pred", "linear"),
}

def compute_metrics(y_true, y_pred):
    r2 = float(r2_score(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    return r2, rmse, mae

# ============ ITEM 1: per-GROUP breakdown, every stratum x property ============
per_group_section = {}
for stratum_name, df in STRATA.items():
    per_group_section[stratum_name] = {}
    groups = sorted(df["GROUP"].dropna().unique())
    for prop_name, (true_col, pred_col, space) in PROPERTIES.items():
        by_group = {}
        suppressed = {}
        for g in groups:
            sub = df[df["GROUP"] == g]
            y_true = sub[true_col].to_numpy(dtype=float)
            y_pred = sub[pred_col].to_numpy(dtype=float)
            n_rows = len(sub)
            n_samples = int(sub["sample_id"].nunique())
            n_clusters = int(sub["chemistry_cluster_id"].nunique())
            if n_samples < MIN_SAMPLES:
                suppressed[g] = {"n_rows": int(n_rows), "n_samples": n_samples, "n_clusters": n_clusters,
                                  "reason": f"n_samples={n_samples} < {MIN_SAMPLES}"}
                continue
            r2, rmse, mae = compute_metrics(y_true, y_pred)
            by_group[g] = {"r2": round(r2, 4), "rmse": round(rmse, 4), "mae": round(mae, 4),
                            "n_rows": int(n_rows), "n_samples": n_samples, "n_clusters": n_clusters}
        per_group_section[stratum_name][prop_name] = {
            "space": space, "by_group": by_group, "suppressed": suppressed,
        }

# stratum-level pooled (all groups) reference, for context
overall_section = {}
for stratum_name, df in STRATA.items():
    overall_section[stratum_name] = {}
    for prop_name, (true_col, pred_col, space) in PROPERTIES.items():
        y_true = df[true_col].to_numpy(dtype=float)
        y_pred = df[pred_col].to_numpy(dtype=float)
        r2, rmse, mae = compute_metrics(y_true, y_pred)
        overall_section[stratum_name][prop_name] = {
            "r2": round(r2, 4), "rmse": round(rmse, 4), "mae": round(mae, 4),
            "n_rows": int(len(df)), "n_samples": int(df["sample_id"].nunique()),
            "n_clusters": int(df["chemistry_cluster_id"].nunique()),
        }

# ============ ITEM 2: SIGMA DIAGNOSTIC in b_and_a_CLEAN ============
def error_share_ranking(df, true_col, pred_col):
    """Per-GROUP sum of squared error and share of total SSE, descending."""
    sub = df.copy()
    sub["sq_err"] = (sub[true_col].astype(float) - sub[pred_col].astype(float)) ** 2
    grp = sub.groupby("GROUP")["sq_err"].sum().sort_values(ascending=False)
    total_sse = grp.sum()
    ranking = []
    cum = 0.0
    for g, sse in grp.items():
        cum += sse
        n_rows = int((sub["GROUP"] == g).sum())
        n_samples = int(sub.loc[sub["GROUP"] == g, "sample_id"].nunique())
        ranking.append({
            "GROUP": g, "sse": round(float(sse), 4),
            "share_of_total_sse": round(float(sse / total_sse), 4),
            "cumulative_share": round(float(cum / total_sse), 4),
            "n_rows": n_rows, "n_samples": n_samples,
        })
    return ranking, float(total_sse)

sigma_ranking, sigma_total_sse = error_share_ranking(df_b_and_a, "sigma_log_true", "sigma_log_pred")

def logo_recompute(df, true_col, pred_col, groups_to_remove):
    sub = df[~df["GROUP"].isin(groups_to_remove)]
    y_true = sub[true_col].to_numpy(dtype=float)
    y_pred = sub[pred_col].to_numpy(dtype=float)
    r2, rmse, mae = compute_metrics(y_true, y_pred)
    return {"r2": round(r2, 4), "rmse": round(rmse, 4), "mae": round(mae, 4),
            "n_rows_remaining": int(len(sub)), "n_samples_remaining": int(sub["sample_id"].nunique()),
            "groups_removed": groups_to_remove}

sigma_baseline_r2 = overall_section["b_and_a_CLEAN"]["sigma_log10"]["r2"]
worst_group = sigma_ranking[0]["GROUP"]
top2_groups = [sigma_ranking[0]["GROUP"], sigma_ranking[1]["GROUP"]] if len(sigma_ranking) > 1 else [sigma_ranking[0]["GROUP"]]

sigma_logo_worst1 = logo_recompute(df_b_and_a, "sigma_log_true", "sigma_log_pred", [worst_group])
sigma_logo_top2 = logo_recompute(df_b_and_a, "sigma_log_true", "sigma_log_pred", top2_groups)

sigma_diagnostic = {
    "label": "LEAVE-ONE/TWO-GROUP-OUT DIAGNOSTIC -- NOT a corrected headline number. "
             "Reported to characterize whether the b_and_a_CLEAN sigma R2=-0.29 failure "
             "is uniform across GROUPs or concentrated in a few.",
    "baseline_stratum_r2": sigma_baseline_r2,
    "baseline_n_rows": overall_section["b_and_a_CLEAN"]["sigma_log10"]["n_rows"],
    "total_sse_log10_space": round(sigma_total_sse, 4),
    "error_share_ranking": sigma_ranking,
    "cumulative_error_share_top1": sigma_ranking[0]["cumulative_share"],
    "cumulative_error_share_top2": sigma_ranking[1]["cumulative_share"] if len(sigma_ranking) > 1 else None,
    "cumulative_error_share_top3": sigma_ranking[2]["cumulative_share"] if len(sigma_ranking) > 2 else None,
    "r2_with_worst_group_removed": sigma_logo_worst1,
    "r2_with_top2_groups_removed": sigma_logo_top2,
}

# ============ ITEM 3: same ranking for S and kappa in b_and_a_CLEAN ============
S_ranking, S_total_sse = error_share_ranking(df_b_and_a, "S_true", "S_pred")
kappa_ranking, kappa_total_sse = error_share_ranking(df_b_and_a, "kappa_log_true", "kappa_log_pred")

comparison_section = {
    "sigma_log10": {
        "total_sse": round(sigma_total_sse, 4),
        "top1_share": sigma_ranking[0]["cumulative_share"],
        "top2_share": sigma_ranking[1]["cumulative_share"] if len(sigma_ranking) > 1 else None,
        "top3_share": sigma_ranking[2]["cumulative_share"] if len(sigma_ranking) > 2 else None,
        "ranking": sigma_ranking,
    },
    "S": {
        "total_sse": round(S_total_sse, 4),
        "top1_share": S_ranking[0]["cumulative_share"],
        "top2_share": S_ranking[1]["cumulative_share"] if len(S_ranking) > 1 else None,
        "top3_share": S_ranking[2]["cumulative_share"] if len(S_ranking) > 2 else None,
        "ranking": S_ranking,
    },
    "kappa_log10": {
        "total_sse": round(kappa_total_sse, 4),
        "top1_share": kappa_ranking[0]["cumulative_share"],
        "top2_share": kappa_ranking[1]["cumulative_share"] if len(kappa_ranking) > 1 else None,
        "top3_share": kappa_ranking[2]["cumulative_share"] if len(kappa_ranking) > 2 else None,
        "ranking": kappa_ranking,
    },
}

# ============ WRITE FINAL JSON ============
output = {
    "note": "Read-only per-GROUP breakdown of already-saved teMatDb per-row predictions "
            "(no refit, no model touch). Sigma/kappa in log10 space, S/zT linear, matching "
            "how each was scored. b_and_a_CLEAN = stratum b (cluster-disjoint) intersected "
            "with stratum a (DOI not in training), i.e. the 25-sample clean chemistry-"
            "transfer subset from corrections.json section K1.",
    "min_samples_for_reporting": MIN_SAMPLES,
    "item1_per_group_breakdown": per_group_section,
    "item1_stratum_overall_reference": overall_section,
    "item2_sigma_diagnostic_b_and_a_CLEAN": sigma_diagnostic,
    "item3_error_share_comparison_b_and_a_CLEAN": comparison_section,
}

with open(f"{RESULTS_DIR}/per_group_breakdown.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"Wrote {RESULTS_DIR}/per_group_breakdown.json")

# ============ READABLE PRINT ============
print()
print("=" * 100)
print("ITEM 1: STRATUM-LEVEL (all GROUPs pooled) REFERENCE")
print("=" * 100)
for stratum_name in STRATA:
    print(f"--- {stratum_name} ---")
    for prop_name in PROPERTIES:
        o = overall_section[stratum_name][prop_name]
        print(f"  {prop_name:26s} R2={o['r2']:+.4f} RMSE={o['rmse']:>10.4f} MAE={o['mae']:>10.4f} "
              f"n_rows={o['n_rows']:>5d} n_samples={o['n_samples']:>3d} n_clusters={o['n_clusters']:>3d}")

print()
print("=" * 100)
print("ITEM 1: PER-GROUP TABLE, stratum b_and_a_CLEAN (25-sample clean chemistry-transfer subset)")
print("=" * 100)
for prop_name in PROPERTIES:
    print(f"--- {prop_name} ({per_group_section['b_and_a_CLEAN'][prop_name]['space']} space) ---")
    bg = per_group_section["b_and_a_CLEAN"][prop_name]["by_group"]
    for g, m in sorted(bg.items(), key=lambda kv: -kv[1]["n_rows"]):
        print(f"  {g:15s} R2={m['r2']:+.4f} RMSE={m['rmse']:>9.4f} MAE={m['mae']:>9.4f} "
              f"n_rows={m['n_rows']:>3d} n_samples={m['n_samples']:>2d} n_clusters={m['n_clusters']:>2d}")
    supp = per_group_section["b_and_a_CLEAN"][prop_name]["suppressed"]
    if supp:
        print("  SUPPRESSED (n_samples<3):")
        for g, s in supp.items():
            print(f"    {g:15s} n_rows={s['n_rows']:>3d} n_samples={s['n_samples']:>2d} n_clusters={s['n_clusters']:>2d}")

print()
print("=" * 100)
print("ITEM 2: SIGMA DIAGNOSTIC, stratum b_and_a_CLEAN (baseline R2 = %.4f, n_rows=%d)" %
      (sigma_diagnostic["baseline_stratum_r2"], sigma_diagnostic["baseline_n_rows"]))
print("=" * 100)
print("Error-share ranking (log10 space):")
for row in sigma_ranking:
    print(f"  {row['GROUP']:15s} SSE={row['sse']:>10.4f}  share={row['share_of_total_sse']:.4f}  "
          f"cumulative={row['cumulative_share']:.4f}  n_rows={row['n_rows']:>3d} n_samples={row['n_samples']:>2d}")
print(f"Cumulative error share: top1={sigma_diagnostic['cumulative_error_share_top1']:.4f}  "
      f"top2={sigma_diagnostic['cumulative_error_share_top2']}  top3={sigma_diagnostic['cumulative_error_share_top3']}")
print()
print("LEAVE-ONE/TWO-GROUP-OUT DIAGNOSTIC (NOT a corrected headline number):")
w = sigma_diagnostic["r2_with_worst_group_removed"]
print(f"  Remove worst GROUP ({worst_group}): R2={w['r2']:+.4f}  n_rows_remaining={w['n_rows_remaining']}  n_samples_remaining={w['n_samples_remaining']}")
t2 = sigma_diagnostic["r2_with_top2_groups_removed"]
print(f"  Remove top-2 GROUPs ({top2_groups}): R2={t2['r2']:+.4f}  n_rows_remaining={t2['n_rows_remaining']}  n_samples_remaining={t2['n_samples_remaining']}")

print()
print("=" * 100)
print("ITEM 3: ERROR-SHARE RANKING COMPARISON, stratum b_and_a_CLEAN -- S, sigma_log10, kappa_log10")
print("=" * 100)
for prop in ["S", "sigma_log10", "kappa_log10"]:
    c = comparison_section[prop]
    print(f"--- {prop} (total SSE={c['total_sse']:.4f}) ---")
    print(f"  top1_share={c['top1_share']:.4f}  top2_share={c['top2_share']}  top3_share={c['top3_share']}")
    for row in c["ranking"][:5]:
        print(f"    {row['GROUP']:15s} share={row['share_of_total_sse']:.4f}  cumulative={row['cumulative_share']:.4f}  n_rows={row['n_rows']}")
