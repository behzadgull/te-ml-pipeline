"""
Smoke test of every Kaggle compute script and of the harness, on a tiny subset (about 3,000 rows, a few trees). Run it on a Linux CPU notebook
before any GPU session; the report it writes is committed as proof, as paper_b did (paper_b/results/smoke_test_linux).

Checks
  1. every script (na7, na2 random forest, na2 LightGBM, na3, na6, na13, final models) runs to completion in --smoke mode and writes status.json
     complete, a manifest whose SHA256 values match the files, and a run_config recording dataset SHA256, git HEAD and tree_clean;
  2. identity enforcement: without --allow-dirty and without --expect-commit a script refuses to start; with a wrong --expect-commit it refuses;
  3. resume: na7 is run with --max-units 1 (incomplete), restored into a second session from the directory (completes, 1 unit restored), restored
     from the second session's .tar.gz into a third (nothing recomputed), and the third session's results equal an uninterrupted run's;
  4. refusal: a restore into a run with different parameters is refused, and so is a restore from a directory with a tampered unit file;
  5. time budget: na2 random forest with --time-budget-hours 0 stops before any unit and reports incomplete, and a restore of that session completes.
When --expect-commit is given the scripts run WITHOUT --allow-dirty (the way the real sessions run) and the tree must be clean.

Usage:
    python thesis_paper/scripts/kaggle/smoke_test_kaggle.py --dataset <snapfix csv> --work-dir <dir> [--expect-commit <sha>] [--allow-dirty] [--jarvis-csv <csv> --jarvis-sha256 <sha>]
"""

import argparse
import json
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import harness as H  # noqa: E402

REPORT = {"checks": {}}


def run(script, *extra, expect_fail=False):
    cmd = [sys.executable, str(HERE / script), *extra]
    t0 = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=H.repo_root())
    ok = (p.returncode != 0) if expect_fail else (p.returncode == 0)
    if not ok:
        print(p.stdout[-3000:], p.stderr[-3000:])
    return ok, p, time.perf_counter() - t0


def status(d):
    return json.loads((Path(d) / "status.json").read_text(encoding="utf-8"))


def manifest_ok(d):
    m = json.loads((Path(d) / "manifest.json").read_text(encoding="utf-8"))
    return all((Path(d) / rel).exists() and H.sha256_file(Path(d) / rel) == sha for rel, sha in m["files"].items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--expect-commit", default=None)
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--jarvis-csv"); ap.add_argument("--jarvis-sha256")
    ap.add_argument("--report", default=None)
    args = ap.parse_args()
    work = Path(args.work_dir)
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    ident = ["--allow-dirty"] if args.allow_dirty else ["--expect-commit", args.expect_commit]
    base = ["--dataset", args.dataset, "--smoke", *ident]

    def check(name, ok, detail=""):
        REPORT["checks"][name] = {"ok": bool(ok), "detail": detail}
        print(("PASS " if ok else "FAIL ") + name, detail, flush=True)

    # 1. every script completes
    jobs = {
        "na7": ("na7_random_dvd.py", []),
        "na2_random_forest": ("na2_trees.py", ["--model", "random_forest", "--targets", "zT"]),
        "na2_lightgbm": ("na2_trees.py", ["--model", "lightgbm", "--targets", "zT"]),
        "na3": ("na3_shap.py", ["--targets", "zT,S"]),
        "na6": ("na6_classifier.py", []),
        "na13": ("na13_feature_selection.py", ["--targets", "zT"]),
        "final_models": ("na_final_models.py", ["--jarvis-csv", args.jarvis_csv, "--jarvis-sha256", args.jarvis_sha256] if args.jarvis_csv else []),
    }
    for name, (script, extra) in jobs.items():
        out = work / name
        ok, p, secs = run(script, "--out-dir", str(out), *base, *extra)
        st = status(out) if ok and (out / "status.json").exists() else {}
        cfg = json.loads(next((out / "run_configs").glob("session_*.json")).read_text(encoding="utf-8")) if ok else {}
        check(f"{name}_completes", ok and st.get("complete"), f"{st.get('units_done')} units in {secs:.0f}s")
        check(f"{name}_manifest_matches_files", ok and manifest_ok(out))
        check(f"{name}_records_identity", ok and bool(cfg.get("dataset_sha256")) and bool(cfg.get("git_head")) and "tree_clean" in cfg,
              f"tree_clean={cfg.get('tree_clean')} allow_dirty={cfg.get('allow_dirty')}")

    # 2. identity enforcement (the script must refuse before touching the data)
    ok, p, _ = run("na7_random_dvd.py", "--out-dir", str(work / "refuse1"), "--dataset", args.dataset, "--smoke", expect_fail=True)
    check("refuses_without_expect_commit_or_allow_dirty", ok)
    if args.expect_commit:
        ok, p, _ = run("na7_random_dvd.py", "--out-dir", str(work / "refuse2"), "--dataset", args.dataset, "--smoke", "--expect-commit", "0" * 40, expect_fail=True)
        check("refuses_wrong_commit", ok)

    # 3. resume chain on na7
    a, b, c, ref = work / "res_a", work / "res_b", work / "res_c", work / "na7"
    ok_a, _, _ = run("na7_random_dvd.py", "--out-dir", str(a), *base, "--max-units", "1")
    check("resume_session_a_incomplete", ok_a and not status(a)["complete"] and status(a)["units_done"] == 1)
    ok_b, _, _ = run("na7_random_dvd.py", "--out-dir", str(b), *base, "--restore-from", str(a))
    sb = status(b)
    check("resume_session_b_completes_with_restored_unit", ok_b and sb["complete"] and sb["units_restored_this_session"] == 1 and sb["units_computed_this_session"] == sb["units_total"] - 1, str(sb))
    ok_c, _, _ = run("na7_random_dvd.py", "--out-dir", str(c), *base, "--restore-from", str(b) + ".tar.gz")
    sc = status(c)
    check("resume_session_c_recomputes_nothing", ok_c and sc["complete"] and sc["units_computed_this_session"] == 0, str(sc))
    same = json.loads((c / "results.json").read_text(encoding="utf-8")) == json.loads((ref / "results.json").read_text(encoding="utf-8"))
    check("resumed_results_equal_uninterrupted_results", same)

    # 3b. two halves (the two-GPU split), merged by a final session
    h0, h1, hm = work / "half0", work / "half1", work / "half_merged"
    ok0, _, _ = run("na7_random_dvd.py", "--out-dir", str(h0), *base, "--shard", "0/2")
    ok1, _, _ = run("na7_random_dvd.py", "--out-dir", str(h1), *base, "--shard", "1/2")
    check("split_halves_each_complete_without_results", ok0 and ok1 and status(h0)["complete"] and status(h1)["complete"]
          and not (h0 / "results.json").exists() and not (h1 / "results.json").exists())
    okm, _, _ = run("na7_random_dvd.py", "--out-dir", str(hm), *base, "--restore-from", f"{h0}.tar.gz,{h1}.tar.gz")
    sm = status(hm) if okm else {}
    check("split_halves_merge_computes_nothing", okm and sm.get("complete") and sm.get("units_computed_this_session") == 0 and sm.get("units_restored_this_session") == sm.get("units_total"), str(sm))
    same = okm and json.loads((hm / "results.json").read_text(encoding="utf-8")) == json.loads((ref / "results.json").read_text(encoding="utf-8"))
    check("split_halves_results_equal_uninterrupted", same)

    # 4. refusals
    ok, _, _ = run("na7_random_dvd.py", "--out-dir", str(work / "res_d"), *base, "--seed", "1", "--restore-from", str(a), expect_fail=True)
    check("restore_refused_on_different_parameters", ok)
    tamper = work / "tampered"
    shutil.copytree(a, tamper)
    npz = next((tamper / "units").glob("*.npz"))
    npz.write_bytes(npz.read_bytes() + b"x")
    ok, _, _ = run("na7_random_dvd.py", "--out-dir", str(work / "res_e"), *base, "--restore-from", str(tamper), expect_fail=True)
    check("restore_refused_on_tampered_unit", ok)

    # 5. time budget
    t1, t2 = work / "tb_a", work / "tb_b"
    ok1, _, _ = run("na2_trees.py", "--out-dir", str(t1), *base, "--model", "random_forest", "--targets", "zT", "--time-budget-hours", "0")
    check("time_budget_stops_before_any_unit", ok1 and not status(t1)["complete"] and status(t1)["units_done"] == 0)
    ok2, _, _ = run("na2_trees.py", "--out-dir", str(t2), *base, "--model", "random_forest", "--targets", "zT", "--restore-from", str(t1))
    check("budget_stopped_session_is_continued", ok2 and status(t2)["complete"])

    REPORT["all_passed"] = all(v["ok"] for v in REPORT["checks"].values())
    REPORT["environment"] = {"python": sys.version.split()[0], "platform": sys.platform, "git_head": H.git_state()[0], "tree_clean": H.git_state()[1],
                             "allow_dirty": args.allow_dirty}
    out = Path(args.report) if args.report else work / "smoke_report.json"
    out.write_text(json.dumps(REPORT, indent=2), encoding="utf-8")
    print(f"all_passed: {REPORT['all_passed']}  report: {out}")
    return 0 if REPORT["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
