"""
Share of the random-to-chemistry inflation that composition grouping removes,
per target: (random - composition) / (random - chemistry cluster), on the
Table 1 rungs. Reproduces the paper's "grouping by exact composition removes
only 58 to 65% of that inflation" (introduction, contribution 1) and the
Seebeck random-to-composition drop of 0.126 (section 4.1). Asserts each
quoted value; run from the repo root.

Inputs (tracked): reports/ungrouped_snapfix/20260922T093243/metrics.json
(random 80/20 pooled R^2) and reports/regen_snapfix/20260917T150000/
ladder_metrics.json (composition and chemistry-cluster per-repeat pooled
R^2 means).
"""

import json

TARGETS = ["S", "sigma", "kappa", "zT"]
EXPECT_SHARE_PCT = {"S": 61, "sigma": 65, "kappa": 65, "zT": 58}


def main():
    with open("reports/ungrouped_snapfix/20260922T093243/metrics.json", encoding="utf-8") as f:
        random_json = json.load(f)["per_run"]
    with open("reports/regen_snapfix/20260917T150000/ladder_metrics.json", encoding="utf-8") as f:
        runs = json.load(f)["runs"]

    shares = {}
    for t in TARGETS:
        rand = random_json[t]["random"]["pooled_r2"]
        comp = runs[f"{t}_composition_full"]["per_repeat_r2_mean"]
        chem = runs[f"{t}_chemistry_full"]["per_repeat_r2_mean"]
        shares[t] = (rand - comp) / (rand - chem)
        print(f"{t:6s} random-composition {rand - comp:.4f}  random-chemistry {rand - chem:.4f}  share {100 * shares[t]:.2f}%")
        assert round(100 * shares[t]) == EXPECT_SHARE_PCT[t], (t, shares[t])
        if t == "S":
            assert round(rand - comp, 3) == 0.126, rand - comp

    lo, hi = min(shares.values()), max(shares.values())
    assert (round(100 * lo), round(100 * hi)) == (58, 65), (lo, hi)
    print(f"range: {100 * lo:.2f}% to {100 * hi:.2f}%  (paper: 58 to 65%)")


if __name__ == "__main__":
    main()
