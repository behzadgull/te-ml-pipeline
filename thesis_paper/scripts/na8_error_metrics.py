"""
NA8: MAE and RMSE per target, and the data behind the predicted-versus-measured figure, from the committed per-row predictions of
the Paper A chemistry-cluster rung (results/ladder_regen_snapfix/20260917T150000) and of the direct-versus-derived run
(results/direct_vs_derived_snapfix/20260924T124138_per_target_cuda). No model is fitted.

Every R2 recomputed from the predictions must equal the committed value (per-repeat pooled R2 for the ladder, pooled R2 for the
direct-versus-derived run) to 1e-9, which ties these predictions to the numbers in the paper. MAE and RMSE are in the space the
target was trained in (log10 for sigma and kappa).

Usage (from the repository root):  python thesis_paper/scripts/na8_error_metrics.py
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import paper_a_values as pav  # noqa: E402
import run_record as rr  # noqa: E402

LADDER_DIR = "results/ladder_regen_snapfix/20260917T150000"
DVD_DIR = "results/direct_vs_derived_snapfix/20260924T124138_per_target_cuda"
T4 = pav.TARGETS
N_REPEATS, N_FOLDS = 5, 5


def r2(y, p):
    return 1.0 - float(np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def metrics(y, p):
    return {"r2": r2(y, p), "mae": float(np.mean(np.abs(y - p))), "rmse": float(np.sqrt(np.mean((y - p) ** 2))), "n": int(y.size)}


def main():
    """Entry point."""
    ladder = pav._json(pav.LADDER)["runs"]
    inputs = {}
    for t in T4:
        for r in range(N_REPEATS):
            for f in range(N_FOLDS):
                inputs[f"{t}_r{r}_f{f}"] = f"{LADDER_DIR}/{t}_chemistry_full/repeat{r}_fold{f}_predictions.npz"
    for r in range(N_REPEATS):
        for f in range(N_FOLDS):
            inputs[f"dvd_r{r}_f{f}"] = f"{DVD_DIR}/repeat{r}_fold{f}.npz"
    inputs["ladder_metrics"] = pav.LADDER
    inputs["dvd_results"] = pav.DVD
    prov = rr.provenance(inputs, __file__)

    res = {"ladder": {}, "dvd": {}}
    parity = {}
    for t in T4:
        per_repeat, repeat0 = [], None
        for r in range(N_REPEATS):
            ys, ps = [], []
            for f in range(N_FOLDS):
                z = np.load(REPO / inputs[f"{t}_r{r}_f{f}"])
                ys.append(z["y_true"].astype(float))
                ps.append(z["y_pred"].astype(float))
            y, p = np.concatenate(ys), np.concatenate(ps)
            per_repeat.append(metrics(y, p))
            if r == 0:
                repeat0 = (y, p)
        committed = ladder[f"{t}_chemistry_full"]["per_repeat_r2"]
        for r in range(N_REPEATS):
            assert abs(per_repeat[r]["r2"] - committed[r]) < 1e-9, (t, r, per_repeat[r]["r2"], committed[r])
        d = {"per_repeat": per_repeat}
        for k in ("r2", "mae", "rmse"):
            v = np.array([m[k] for m in per_repeat])
            d[f"{k}_mean"], d[f"{k}_sd"] = float(v.mean()), float(v.std(ddof=1))
        y, p = repeat0
        slope = float(np.polyfit(y, p, 1)[0])
        top = y >= np.quantile(y, 0.95)
        d["repeat0_calibration"] = {"slope_pred_on_true": slope, "top5pct_true_mean": float(y[top].mean()), "top5pct_pred_mean": float(p[top].mean()),
                                    "bottom5pct_true_mean": float(y[y <= np.quantile(y, 0.05)].mean()),
                                    "bottom5pct_pred_mean": float(p[y <= np.quantile(y, 0.05)].mean())}
        res["ladder"][t] = d
        parity[t] = (y, p)
    # parity figure data: 2-D histograms of repeat 0 (counts, not rows), so the figure can be drawn from the committed result
    res["parity_hist2d"] = {}
    for t, (y, p) in parity.items():
        lo, hi = float(min(y.min(), p.min())), float(max(y.max(), p.max()))
        h, xe, ye = np.histogram2d(y, p, bins=80, range=[[lo, hi], [lo, hi]])
        res["parity_hist2d"][t] = {"range": [lo, hi], "counts": h.astype(int).tolist(), "r2_repeat0": r2(y, p), "mae_repeat0": float(np.mean(np.abs(y - p)))}

    # direct versus derived zT, pooled over all folds exactly as the committed pooled R2
    arrs = {k: [] for k in ("zT_direct_true", "zT_direct_pred", "zT_derived_true", "zT_derived_pred")}
    for r in range(N_REPEATS):
        for f in range(N_FOLDS):
            z = np.load(REPO / inputs[f"dvd_r{r}_f{f}"])
            for k in arrs:
                arrs[k].append(z[k].astype(float))
    arrs = {k: np.concatenate(v) for k, v in arrs.items()}
    dvd = pav._json(pav.DVD)
    for name, key in (("direct", "zT_direct"), ("derived", "zT_derived")):
        m = metrics(arrs[f"{key}_true"], arrs[f"{key}_pred"])
        assert abs(m["r2"] - dvd[key]["pooled_r2"]) < 1e-9, (name, m["r2"], dvd[key]["pooled_r2"])
        assert m["n"] == dvd[key]["n"]
        res["dvd"][name] = m

    out = rr.new_run_dir("na8")
    rr.write_json(out / "run_config.json", prov)
    rr.write_json(out / "results.json", res)
    print(f"wrote {out}")
    for t in T4:
        d = res["ladder"][t]
        c = d["repeat0_calibration"]
        print(t, f"R2 {d['r2_mean']:.4f}  MAE {d['mae_mean']:.4f} +- {d['mae_sd']:.4f}  RMSE {d['rmse_mean']:.4f} +- {d['rmse_sd']:.4f}  slope {c['slope_pred_on_true']:.3f}",
              f"top5%: true {c['top5pct_true_mean']:.3f} pred {c['top5pct_pred_mean']:.3f}")
    print("dvd", res["dvd"])


if __name__ == "__main__":
    main()
