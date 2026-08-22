"""
Back-transform bias diagnostic for the direct-vs-derived zT comparison
(CLAUDE.md Paper A item 5).

sigma and kappa are trained/predicted in log10 space (LOG_TRANSFORM_
TARGETS, see src/nested_cv.py), then exponentiated back to linear scale
to form the derived pathway's S^2*sigma*T/kappa. Naive exponentiation of
a log-scale prediction is a biased estimator of the linear-scale
conditional mean (Jensen's inequality: E[10^X] > 10^E[X] for any X with
nonzero spread), so this module checks whether that back-transform bias
-- rather than genuine multiplicative error propagation across three
models -- explains the R^2 gap between the direct and derived pathways
found in src/direct_vs_derived_zt.py (0.7931 vs 0.6659).

Three checks, run on the pooled predictions from that module's saved
per-fold checkpoints (checkpoints/direct_vs_derived_zt/):
1. Residual distribution shape (direct vs derived) -- a back-transform
   artifact should show as a concentrated outlier tail (occasional huge
   errors from small predicted-kappa in the denominator); genuine error
   propagation should show as a broadly wider distribution instead.
2. Duan (1983) smearing correction applied to sigma/kappa's back-
   transform, derived-zT R^2 with vs without.
3. Residual correlation matrix across S, sigma (log10), kappa (log10) --
   CLAUDE.md's item 5 requires reporting this rather than assuming
   independence.

Reconstructs the exact subset/splits src/direct_vs_derived_zt.py used
(same seed, same local featurized CSV) to recover each pooled
prediction's temperature_bin -- not saved in the checkpoint npz files --
and VERIFIES the reconstruction against the checkpoint's own saved
*_true arrays before trusting it (same discipline as scripts/
make_figures.py's Figure 3 checkpoint verification). This run is fully
local (unlike Figure 3's Kaggle checkpoints), so it verifies exactly.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.direct_vs_derived_zt import CHECKPOINT_DIR, load_all_four_subset
from src.nested_cv import GROUP_COL, N_OUTER_FOLDS, N_OUTER_REPEATS_GROUPED, randomized_group_kfold
from src.plotting_style import COLORBLIND_PALETTE, add_panel_label, apply_style, get_figsize, save_figure

FIGURES_DIR = Path("figures")
SEED = 0  # matches direct_vs_derived_zt.run_direct_vs_derived's default


def load_pooled_verified_data(checkpoint_dir=CHECKPOINT_DIR, seed=SEED):
    """
    Regenerate the exact subset/fold splits run_direct_vs_derived() used,
    verify each fold's reconstructed S/sigma_log10/kappa_log10/zT against
    that fold's checkpointed *_true arrays (exact elementwise match
    required), then return the pooled arrays plus each row's
    temperature_bin (recovered this way since it isn't itself saved in
    the checkpoint npz files).

    Raises AssertionError if any fold fails to verify -- refuses to
    return temperature data that hasn't been confirmed correct rather
    than silently mislabeling it.
    """
    subset, source_path = load_all_four_subset()
    groups = subset[GROUP_COL].to_numpy()
    T_all = subset["temperature_bin"].to_numpy(dtype=np.float64)
    S_all = subset["S"].to_numpy(dtype=np.float64)
    sigma_log_all = np.log10(subset["sigma"].to_numpy(dtype=np.float64))
    kappa_log_all = np.log10(subset["kappa"].to_numpy(dtype=np.float64))
    zT_all = subset["zT"].to_numpy(dtype=np.float64)

    rng_master = np.random.default_rng(seed)
    pooled = {
        "S_true": [], "S_pred": [],
        "sigma_log_true": [], "sigma_log_pred": [],
        "kappa_log_true": [], "kappa_log_pred": [],
        "zT_direct_true": [], "zT_direct_pred": [],
        "zT_derived_true": [], "zT_derived_pred": [],
        "T": [],
    }

    n_verified, n_total = 0, 0
    for repeat in range(N_OUTER_REPEATS_GROUPED):
        repeat_rng = np.random.default_rng(rng_master.integers(0, 2**32 - 1))
        fold_iter = randomized_group_kfold(groups, N_OUTER_FOLDS, repeat_rng)
        for fold, (train_idx, test_idx) in enumerate(fold_iter):
            n_total += 1
            data = np.load(Path(checkpoint_dir) / f"repeat{repeat}_fold{fold}.npz")

            ok = (
                np.allclose(S_all[test_idx], data["S_true"], rtol=1e-8, atol=1e-8)
                and np.allclose(sigma_log_all[test_idx], data["sigma_log10_true"], rtol=1e-8, atol=1e-8)
                and np.allclose(kappa_log_all[test_idx], data["kappa_log10_true"], rtol=1e-8, atol=1e-8)
                and np.allclose(zT_all[test_idx], data["zT_direct_true"], rtol=1e-8, atol=1e-8)
            )
            if not ok:
                raise AssertionError(
                    f"repeat {repeat} fold {fold}: local reconstruction does not match the "
                    f"checkpointed *_true arrays -- refusing to trust temperature_bin recovery"
                )
            n_verified += 1

            pooled["S_true"].append(data["S_true"]); pooled["S_pred"].append(data["S_pred"])
            pooled["sigma_log_true"].append(data["sigma_log10_true"]); pooled["sigma_log_pred"].append(data["sigma_log10_pred"])
            pooled["kappa_log_true"].append(data["kappa_log10_true"]); pooled["kappa_log_pred"].append(data["kappa_log10_pred"])
            pooled["zT_direct_true"].append(data["zT_direct_true"]); pooled["zT_direct_pred"].append(data["zT_direct_pred"])
            pooled["zT_derived_true"].append(data["zT_derived_true"]); pooled["zT_derived_pred"].append(data["zT_derived_pred"])
            pooled["T"].append(T_all[test_idx])

    print(f"Verified {n_verified}/{n_total} folds against {source_path}")
    return {k: np.concatenate(v).astype(np.float64) for k, v in pooled.items()}


def residual_distribution_stats(direct_resid, derived_resid):
    """Mean/std/percentiles and tail-SSE-share for both residual arrays -- see module docstring, check 1."""
    stats = {}
    for name, r in [("direct", direct_resid), ("derived", derived_resid)]:
        n = len(r)
        sse = float(np.sum(r ** 2))
        order = np.argsort(-np.abs(r))
        top1 = order[: max(1, n // 100)]
        top5 = order[: max(1, n // 20)]
        stats[name] = {
            "mean": float(r.mean()), "std": float(r.std()), "median": float(np.median(r)),
            "p1": float(np.percentile(r, 1)), "p99": float(np.percentile(r, 99)),
            "max_abs": float(np.abs(r).max()),
            "sse": sse,
            "top1pct_sse_share": float(np.sum(r[top1] ** 2) / sse),
            "top5pct_sse_share": float(np.sum(r[top5] ** 2) / sse),
        }
    return stats


def duan_smearing_correction(d):
    """
    Apply Duan (1983) smearing to sigma/kappa's back-transform: multiply
    each 10^(log_pred) by mean(10^(held-out log residual)) before
    re-forming derived zT = (S_pred/1e6)^2 * sigma_pred * T / kappa_pred.
    Returns (zT_derived_pred_corrected, smear_sigma, smear_kappa).
    """
    smear_sigma = float(np.mean(10.0 ** (d["sigma_log_true"] - d["sigma_log_pred"])))
    smear_kappa = float(np.mean(10.0 ** (d["kappa_log_true"] - d["kappa_log_pred"])))

    sigma_pred_corrected = (10.0 ** d["sigma_log_pred"]) * smear_sigma
    kappa_pred_corrected = (10.0 ** d["kappa_log_pred"]) * smear_kappa

    zT_derived_pred_corrected = (
        ((d["S_pred"] / 1.0e6) ** 2) * sigma_pred_corrected * d["T"] / kappa_pred_corrected
    )
    return zT_derived_pred_corrected, smear_sigma, smear_kappa


def residual_correlation_matrix(d):
    """Pearson correlation matrix across S (linear), sigma (log10), kappa (log10) residuals."""
    S_resid = d["S_true"] - d["S_pred"]
    sigma_log_resid = d["sigma_log_true"] - d["sigma_log_pred"]
    kappa_log_resid = d["kappa_log_true"] - d["kappa_log_pred"]
    labels = ["S", "sigma (log10)", "kappa (log10)"]
    corr = np.corrcoef(np.vstack([S_resid, sigma_log_resid, kappa_log_resid]))
    return corr, labels


def make_residual_distribution_figure(direct_resid, derived_resid, out_path):
    """
    Figure: two-panel histogram of direct vs. derived zT residuals --
    full range (a) and central 99% (b), overlaid. A back-transform
    artifact would show derived with a disproportionate outlier tail
    relative to direct; genuine error propagation shows a broadly wider
    distribution instead (see module docstring, check 1).
    """
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=get_figsize(1, 2, panel_width=5.0, panel_height=4.2))

    bins = np.linspace(
        min(direct_resid.min(), derived_resid.min()),
        max(direct_resid.max(), derived_resid.max()),
        120,
    )
    axes[0].hist(direct_resid, bins=bins, color=COLORBLIND_PALETTE[5], alpha=0.6, label="Direct", density=True)
    axes[0].hist(derived_resid, bins=bins, color=COLORBLIND_PALETTE[6], alpha=0.6, label="Derived", density=True)
    axes[0].set_xlabel("Residual (actual $-$ predicted zT)")
    axes[0].set_ylabel("Density")
    axes[0].legend()
    add_panel_label(axes[0], "a")

    lo, hi = np.percentile(np.concatenate([direct_resid, derived_resid]), [0.5, 99.5])
    axes[1].hist(
        direct_resid, bins=120, range=(lo, hi), color=COLORBLIND_PALETTE[5],
        alpha=0.6, label="Direct", density=True,
    )
    axes[1].hist(
        derived_resid, bins=120, range=(lo, hi), color=COLORBLIND_PALETTE[6],
        alpha=0.6, label="Derived", density=True,
    )
    axes[1].set_xlabel("Residual (actual $-$ predicted zT), central 99%")
    axes[1].set_ylabel("Density")
    axes[1].legend()
    add_panel_label(axes[1], "b")

    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


def report(d, stats, r2_direct, r2_derived_uncorrected, r2_derived_corrected, smear_sigma, smear_kappa, corr, labels):
    """Print all three diagnostics' findings in a readable form."""
    n = len(d["zT_direct_true"])
    print(f"\nn = {n:,}")
    print(f"Direct zT R^2 = {r2_direct:.4f}")
    print(f"Derived zT R^2 (uncorrected) = {r2_derived_uncorrected:.4f}, gap = {r2_direct - r2_derived_uncorrected:.4f}")

    print("\n=== 1. Residual distribution ===")
    for name in ("direct", "derived"):
        s = stats[name]
        print(
            f"{name:8s} mean={s['mean']:+.4f} std={s['std']:.4f} p1={s['p1']:+.4f} p99={s['p99']:+.4f} "
            f"max_abs={s['max_abs']:.4f} top1%SSEshare={s['top1pct_sse_share']:.1%} "
            f"top5%SSEshare={s['top5pct_sse_share']:.1%}"
        )

    print("\n=== 2. Duan smearing correction ===")
    print(f"smear_sigma={smear_sigma:.4f}  smear_kappa={smear_kappa:.4f}")
    print(f"Derived zT R^2 WITHOUT correction = {r2_derived_uncorrected:.4f}")
    print(f"Derived zT R^2 WITH correction    = {r2_derived_corrected:.4f}")

    print("\n=== 3. Residual correlation matrix ===")
    print(f"{'':16s}" + "".join(f"{l:>16s}" for l in labels))
    for i, l in enumerate(labels):
        print(f"{l:16s}" + "".join(f"{corr[i, j]:16.4f}" for j in range(len(labels))))

    verdict = "SURVIVES -- genuine error propagation" if r2_derived_corrected <= r2_derived_uncorrected + 1e-6 else "COLLAPSES -- back-transform artifact"
    print(f"\nConclusion: Duan correction result -> gap {verdict}")


def main():
    d = load_pooled_verified_data()

    direct_resid = d["zT_direct_true"] - d["zT_direct_pred"]
    derived_resid = d["zT_derived_true"] - d["zT_derived_pred"]
    stats = residual_distribution_stats(direct_resid, derived_resid)

    r2_direct = r2_score(d["zT_direct_true"], d["zT_direct_pred"])
    r2_derived_uncorrected = r2_score(d["zT_derived_true"], d["zT_derived_pred"])
    zT_derived_pred_corrected, smear_sigma, smear_kappa = duan_smearing_correction(d)
    r2_derived_corrected = r2_score(d["zT_derived_true"], zT_derived_pred_corrected)

    corr, labels = residual_correlation_matrix(d)

    report(d, stats, r2_direct, r2_derived_uncorrected, r2_derived_corrected, smear_sigma, smear_kappa, corr, labels)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    make_residual_distribution_figure(direct_resid, derived_resid, FIGURES_DIR / "fig_backtransform_check")
    print(f"\nSaved fig_backtransform_check.png / .pdf to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
