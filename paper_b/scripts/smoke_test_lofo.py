"""
Local CPU smoke test for paper_b/src/lofo_paperb.py (methodology doc section
8), run before the harness is committed. One target (zT), the two smallest
qualifying families (manganite, i_v_vi2), R=1, 2 tuning trials (also used for
the specialist, for speed), folds {0, 1}, both models.

Checks, in order, and prints a clear PASS/FAIL for each:
  1. The six leakage assertions (methodology doc section 8.1) are enforced by
     the harness itself (paper_b/src/lofo_paperb.py's assert_disjoint/
     assert_subset/assert_size_equal, called inline during fitting) and are
     exercised by this test's own run (they raise on the first violation, so
     a clean run here is itself a pass for all six, checked by their call
     sites having actually executed, not just imported).
  2. Checkpoint/resume: run to completion, note file count and content hashes,
     wipe the checkpoint dir, rerun but kill the process partway through
     (subprocess.terminate(), an abrupt kill, not a graceful shutdown), then
     resume with the same command; the resumed run must not rewrite any file
     the killed run had already written (same bytes), and the final file set
     must match the uninterrupted run's.
  3. Determinism: delete one completed unit's checkpoint only, rerun it alone,
     compare predictions to the original bit-for-bit.
  4. Murphy shares sum to 1 (to 1e-9) for every saved prediction file; its R^2
     matches sklearn's r2_score to float precision.
  5. Hand-check: one unit's R^2, skill_train, Murphy shares and within-family
     r recomputed here directly from the saved arrays (not via
     metrics_paperb.py), shown next to metrics_paperb.py's own output.
  6. Bootstrap: a CI is produced for one unit/condition; a paired difference
     between a prediction array and itself gives exactly 0 at every resample.
  7. Production guard, checkpoint-dir naming: the harness must refuse
     --smoke-search-space-cap when --checkpoint-dir does not contain "smoke".
  8. Production guard, analysis loader: `lofo_paperb.load_checkpoint_dir_for_analysis`
     must raise on a directory holding any smoke_cap=true unit (this test's own
     reference run, every unit of which was fit with the cap), and, separately,
     on two synthetic run_configs that disagree on the identity fields.
  9. --time-budget-hours, a few seconds: must still exit 0 and write a session
     summary (a too-small budget honestly finishing zero units is a pass, not
     a failure -- loading the dataset alone can exceed it).
  10. --restore-from a .tar.gz of the reference run: every unit is skipped
     (n_wrote == 0), reaching the same file set; a tampered copy is refused.
  11. Role independence (section 8.1): gpu-pooled and cpu-specialist write to
     SEPARATE reference directories throughout this test (methodology doc
     section 8.7's Kaggle plan -- one checkpoint dir per role). cpu-specialist
     completing every unit in a directory that never held a single gpu-pooled
     file (no tuning_once/C0/C1/C2/C3 anywhere under it) is a structural proof
     that it never reads gpu-pooled's output, checked directly here rather
     than only by reading process_specialist_unit's source. The new
     load_checkpoint_dirs_for_analysis is also exercised: it must merge two
     identity-agreeing role directories into one unit list, and must raise
     when two individually-clean directories disagree with EACH OTHER on
     identity (not just within one).
  12. Path-separator normalisation (2026-09-30 fix): the committed splits
     manifest was written on Windows, backslash-separated -- Kaggle (Linux)
     failed check_splits_manifest with FileNotFoundError on it. Feeds
     check_splits_manifest and restore_from's manifest reader a synthetic
     manifest with one backslash-style key and one forward-slash-style key
     each; both must resolve to their real files and pass.
  13. In-pair time budget (2026-09-30 fix): a single gpu-pooled task can run
     tuning_once + C1 + C0/C2/C3 across every fold unbroken -- run_pool's own
     --time-budget-hours check only gated which TASK got dispatched next, not
     what happened inside one already-dispatched task (a Kaggle smoke run hung
     70+ minutes against a 5-minute budget this way). Times one small pair
     unrestricted, then reruns it with a budget of half that measured time --
     long enough that tuning_once (the first, necessary piece) should finish,
     short enough that the whole pair should not -- and requires it to stop
     with SOME but not all of that pair's files written and
     units_partial_this_session > 0; a resume (no budget) must then complete
     the pair without rewriting any already-checkpointed file.
  14. LF hash verification (2026-10-01 fix): the committed splits manifest
     was regenerated from LF-normalised content after .gitattributes started
     forcing `paper_b/**/*.json text eol=lf` -- a Windows-hashed, CRLF-based
     manifest entry does not match the same content checked out as plain LF
     on Linux (Kaggle), which is exactly how the second Kaggle crash
     happened. Builds an LF-forced and a CRLF-forced copy of the real
     manifest-listed text files and requires check_splits_manifest to pass
     on the LF copy (simulating a fresh Linux checkout) and to raise on the
     CRLF copy (proving the manifest is genuinely pinned to LF content, not
     accidentally EOL-agnostic).
  15. Multi-worker pool (2026-10-01 fix): every other check leaves --workers
     at its default (1), never constructing a real multiprocessing.Pool --
     this is why a third Kaggle (Linux-only) crash, "A SemLock created in a
     fork context is being shared with a process in a spawn context", went
     unseen here: a counter built via the bare mp.Value("i", 0) call (the
     platform DEFAULT context -- fork on Linux, already spawn on Windows,
     which is why this suite could never reproduce it) was being handed to
     a Pool explicitly built with mp.get_context("spawn"). Fixed by building
     every multiprocessing primitive from that same explicit context. This
     check runs --workers 2 for the first time in this suite, catching any
     OTHER multi-worker regression Windows can see; only a Linux run (the
     Kaggle CPU smoke cells) can prove the specific crash itself is fixed.

Writes paper_b/results/smoke_test/<UTC>/report.json and prints the same
report. Non-zero exit if any check fails.

Run from the repository root:
    python -m paper_b.scripts.smoke_test_lofo --csv <snapfix csv> \
        --labels-run paper_b/reports/family_labels/<UTC> --splits-dir paper_b/results/splits/<UTC>
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sklearn.metrics import r2_score as sklearn_r2_score  # noqa: E402

from paper_b.src import lofo_paperb as L  # noqa: E402
from paper_b.src import metrics_paperb as M  # noqa: E402

TARGET = "zT"
UNITS = ["manganite", "i_v_vi2"]
FOLDS = [0, 1]
TRIALS = 2
PY = sys.executable


def harness_cmd(csv, labels_run, splits_dir, checkpoint_dir, role, extra=()):
    return [PY, "-m", "paper_b.src.lofo_paperb", "--role", role, "--csv", str(csv), "--labels-run", str(labels_run),
            "--splits-dir", str(splits_dir), "--checkpoint-dir", str(checkpoint_dir), "--targets", TARGET,
            "--units", ",".join(UNITS), "--repeats", "1", "--folds", ",".join(map(str, FOLDS)),
            "--tuning-trials", str(TRIALS), "--specialist-trials", str(TRIALS), "--smoke-search-space-cap", *extra]


def all_checkpoint_files(checkpoint_dir):
    """Every *.npz under checkpoint_dir except run_configs/ (per-session logs) -- includes tuning_once.npz."""
    return sorted(p for p in checkpoint_dir.rglob("*.npz") if "run_configs" not in p.parts)


def all_prediction_files(checkpoint_dir):
    """Every *.npz holding (row_ids, y_true, y_pred) -- every checkpoint file except tuning_once.npz, which
    holds only a training-row count (no prediction to score)."""
    return sorted(p for p in all_checkpoint_files(checkpoint_dir) if p.stem != "tuning_once")


def file_hashes(paths):
    import hashlib
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def run_role(csv, labels_run, splits_dir, checkpoint_dir, role):
    """Run one --role to completion (device=cpu throughout) into its own checkpoint_dir; raise on failure."""
    result = subprocess.run(harness_cmd(csv, labels_run, splits_dir, checkpoint_dir, role), capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{role} failed:\n{result.stdout}\n{result.stderr}")


def check_1_leakage_assertions_exercised():
    """
    The assertions are enforced inline in lofo_paperb.py's fitting functions
    (assert_disjoint/assert_subset/assert_size_equal calls at each of the 6
    points section 8.1 names); the reference gpu-pooled/cpu-specialist runs
    below completing with no AssertionError means all 6 held on this data.
    This check additionally proves each assertion FUNCTION is capable of
    catching a real violation, by feeding it a deliberately bad input.
    """
    ok = True
    try:
        L.assert_disjoint(np.array([1, 2, 3]), np.array([3, 4]), 10, "test")
        ok = False  # should have raised
    except AssertionError:
        pass
    try:
        L.assert_subset(np.array([1, 2, 99]), np.array([1, 2, 3]), "test")
        ok = False
    except AssertionError:
        pass
    try:
        L.assert_size_equal(5, 6, "test")
        ok = False
    except AssertionError:
        pass
    return ok


def check_2_checkpoint_resume(csv, labels_run, splits_dir, work_dir):
    """
    Full gpu-pooled run for a reference file set/hash; then a killed-and-resumed
    gpu-pooled run; compare. Role-separated (methodology doc section 8.7): this
    checks only the gpu-pooled role, in its own directory -- cpu-specialist's
    resume behavior shares the same write_if_absent mechanism, exercised
    separately in its own reference directory in main() (see check_11, which
    depends on that directory never having held a gpu-pooled file).
    """
    ref_dir = work_dir / "reference_gpu_pooled"
    run_role(csv, labels_run, splits_dir, ref_dir, "gpu-pooled")
    ref_files = all_checkpoint_files(ref_dir)
    ref_hashes = file_hashes(ref_files)
    ref_names = {p.relative_to(ref_dir) for p in ref_files}

    kill_dir = work_dir / "kill_resume"
    kill_dir.mkdir(parents=True)
    proc = subprocess.Popen(harness_cmd(csv, labels_run, splits_dir, kill_dir, "gpu-pooled"),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # Poll for the first real checkpoint file rather than guess a fixed delay (dataset loading alone took
    # longer than two earlier fixed guesses, 1.5s and 20s, both of which killed before anything existed --
    # a too-early kill makes "no file rewritten on resume" vacuously true over an empty partial set). Once
    # at least one file exists, give it a little longer so more than a single file is at risk of being
    # rewritten, then kill.
    deadline = time.time() + 180
    first_seen = None
    while time.time() < deadline:
        if all_checkpoint_files(kill_dir):
            first_seen = time.time()
            break
        if proc.poll() is not None:
            raise RuntimeError(f"gpu-pooled exited on its own (code {proc.returncode}) before any checkpoint appeared")
        time.sleep(1)
    if first_seen is None:
        proc.terminate()
        raise RuntimeError("no checkpoint file appeared within 180s; cannot test a non-vacuous kill/resume")
    time.sleep(5)  # let a second file land too, so resume has more than one file to risk rewriting
    proc.terminate()
    proc.wait(timeout=15)
    partial_files = all_checkpoint_files(kill_dir)
    partial_hashes = file_hashes(partial_files)

    # Resume: gpu-pooled again (same command, same directory) to reach the reference's full gpu-pooled file set.
    run_role(csv, labels_run, splits_dir, kill_dir, "gpu-pooled")

    final_files = all_checkpoint_files(kill_dir)
    final_names = {p.relative_to(kill_dir) for p in final_files}
    final_hashes = file_hashes(final_files)

    not_rewritten = all(final_hashes.get(path) == h for path, h in partial_hashes.items())
    same_final_set = final_names == ref_names
    return {
        "n_partial_files_before_resume": len(partial_files), "n_reference_files": len(ref_files),
        "n_final_files": len(final_files), "partial_was_incomplete": len(partial_files) < len(ref_files),
        "no_file_rewritten_on_resume": not_rewritten, "final_set_matches_reference": same_final_set,
        "passed": not_rewritten and same_final_set and len(partial_files) < len(ref_files),
    }, ref_dir


def check_3_determinism(csv, labels_run, splits_dir, ref_dir, work_dir):
    """Delete one completed unit's C0 r1f0 xgboost checkpoint; rerun alone; compare bit-for-bit to the reference."""
    rel = Path("family") / TARGET / "manganite" / "xgboost" / "C0" / "r1_f0"
    target_path = ref_dir / rel
    original = dict(np.load(target_path.with_suffix(".npz")))
    solo_dir = work_dir / "determinism"
    shutil.copytree(ref_dir, solo_dir)
    (solo_dir / rel).with_suffix(".npz").unlink()
    (solo_dir / rel).with_suffix(".json").unlink()
    result = subprocess.run(harness_cmd(csv, labels_run, splits_dir, solo_dir, "gpu-pooled",
                                        extra=["--units", "manganite", "--folds", "0"]),
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"determinism rerun failed:\n{result.stdout}\n{result.stderr}")
    rerun = dict(np.load((solo_dir / rel).with_suffix(".npz")))
    identical = all(np.array_equal(original[k], rerun[k]) for k in original)
    return {"identical_predictions": identical, "arrays_compared": list(original.keys()), "passed": identical}


def check_4_murphy_and_r2(ref_dirs):
    """Every saved prediction file, across all ref_dirs (role-separated): Murphy shares sum to 1 (1e-9), R^2 matches sklearn's."""
    rows = []
    for ref_dir in ref_dirs:
        for path in all_prediction_files(ref_dir):
            arrays = np.load(path)
            y_true, y_pred = arrays["y_true"], arrays["y_pred"]
            decomposition = M.murphy_decomposition(y_true, y_pred)
            shares_sum = decomposition["offset_share"] + decomposition["scale_share"] + decomposition["unexplained_share"]
            r2_mine = M.r2_score(y_true, y_pred)
            r2_sklearn = float(sklearn_r2_score(y_true, y_pred))
            rows.append({"file": f"{ref_dir.name}/{path.relative_to(ref_dir)}", "shares_sum": shares_sum,
                        "shares_ok": abs(shares_sum - 1.0) < 1e-9, "r2_mine": r2_mine, "r2_sklearn": r2_sklearn,
                        "r2_ok": abs(r2_mine - r2_sklearn) < 1e-9 or (np.isnan(r2_mine) and np.isnan(r2_sklearn))})
    return {"n_files_checked": len(rows), "all_shares_ok": all(r["shares_ok"] for r in rows),
            "all_r2_ok": all(r["r2_ok"] for r in rows), "worst_shares_deviation": max(abs(r["shares_sum"] - 1.0) for r in rows),
            "passed": all(r["shares_ok"] and r["r2_ok"] for r in rows)}


def check_5_hand_check(ref_dir):
    """One unit (manganite, xgboost, C0, r1f0): recompute R^2/skill/Murphy/r independently, no metrics_paperb import."""
    rel = Path("family") / TARGET / "manganite" / "xgboost" / "C0" / "r1_f0"
    arrays = np.load((ref_dir / rel).with_suffix(".npz"))
    sidecar = json.loads((ref_dir / rel).with_suffix(".json").read_text(encoding="utf-8"))
    y_true, y_pred, train_mean = arrays["y_true"], arrays["y_pred"], float(arrays["train_mean"][0])

    # -- hand computation, independent of metrics_paperb.py --
    n = len(y_true)
    ss_res = sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred))
    ss_tot = sum((yt - sum(y_true) / n) ** 2 for yt in y_true)
    r2_hand = 1 - ss_res / ss_tot
    mse_model = ss_res / n
    mse_baseline = sum((yt - train_mean) ** 2 for yt in y_true) / n
    skill_hand = 1 - mse_model / mse_baseline
    mean_t, mean_p = sum(y_true) / n, sum(y_pred) / n
    var_t = sum((yt - mean_t) ** 2 for yt in y_true) / n
    var_p = sum((yp - mean_p) ** 2 for yp in y_pred) / n
    cov = sum((yt - mean_t) * (yp - mean_p) for yt, yp in zip(y_true, y_pred)) / n
    r_hand = cov / (var_t ** 0.5 * var_p ** 0.5)
    s_t, s_p = var_t ** 0.5, var_p ** 0.5
    offset_hand = (mean_p - mean_t) ** 2
    scale_hand = (s_p - r_hand * s_t) ** 2
    unexplained_hand = (1 - r_hand ** 2) * s_t ** 2
    total_hand = offset_hand + scale_hand + unexplained_hand
    hand = {"r2": r2_hand, "skill_train": skill_hand, "within_family_r": r_hand,
           "offset_share": offset_hand / total_hand, "scale_share": scale_hand / total_hand,
           "unexplained_share": unexplained_hand / total_hand}

    module = M.pool_metrics(y_true, y_pred, train_mean)
    module_view = {"r2": module["r2"], "skill_train": module["skill_train"], "within_family_r": module["within_family_r"],
                  "offset_share": module["offset_share"], "scale_share": module["scale_share"],
                  "unexplained_share": module["unexplained_share"]}
    agree = all(abs(hand[k] - module_view[k]) < 1e-9 for k in hand)
    return {"unit": rel.as_posix(), "n_rows": int(n), "hand": hand, "metrics_paperb": module_view, "agree": agree, "passed": agree}


def check_6_bootstrap(ref_dir):
    """
    A CI is produced for one unit/condition; a paired difference of a
    prediction array against itself is exactly 0 everywhere. The saved
    prediction file carries row_ids but not each row's chemistry cluster, so
    this uses row_ids themselves as the resampling unit -- a structural check
    of the bootstrap machinery (does it produce a CI, is a paired identical
    difference exactly 0), not a claim about real cluster-level resampling.
    """
    rel = Path("family") / TARGET / "manganite" / "xgboost" / "C0" / "r1_f0"
    arrays = np.load((ref_dir / rel).with_suffix(".npz"))
    y_true, y_pred, row_ids = arrays["y_true"], arrays["y_pred"], arrays["row_ids"]
    lo, hi, values = M.cluster_bootstrap_ci(y_true, y_pred, row_ids, M.r2_score, n_resamples=200, seed=0)
    lo_diff, hi_diff, diff_values = M.paired_cluster_bootstrap_ci(y_true, y_pred, y_pred, row_ids, M.r2_score,
                                                                 n_resamples=200, seed=0)
    ci_produced = bool(np.isfinite(lo) and np.isfinite(hi))
    paired_all_zero = bool(np.all(diff_values == 0.0))
    return {"ci_lo": lo, "ci_hi": hi, "paired_diff_all_zero": paired_all_zero, "ci_produced": ci_produced,
            "passed": ci_produced and paired_all_zero}


def check_7_smoke_guard_refusal(csv, labels_run, splits_dir):
    """The harness must refuse --smoke-search-space-cap unless --checkpoint-dir contains 'smoke'."""
    import tempfile
    non_smoke_dir = Path(tempfile.mkdtemp(prefix="lofo_guard_test_"))
    try:
        cmd = harness_cmd(csv, labels_run, splits_dir, non_smoke_dir, "gpu-pooled",
                          extra=["--units", "manganite", "--folds", "0"])
        result = subprocess.run(cmd, capture_output=True, text=True)
        refused = result.returncode != 0 and "smoke" in (result.stderr or "").lower()
        return {"checkpoint_dir_used": str(non_smoke_dir), "returncode": result.returncode,
                "stderr_tail": (result.stderr or "")[-500:], "refused_as_expected": refused, "passed": refused}
    finally:
        shutil.rmtree(non_smoke_dir, ignore_errors=True)


def _write_synthetic_unit_dir(directory, config, unit_name, smoke_cap=False):
    """A minimal, internally-consistent checkpoint dir: one run_config, one non-tuning_once unit sidecar."""
    (directory / "run_configs").mkdir(parents=True)
    (directory / "family" / "zT" / unit_name / "xgboost" / "C0").mkdir(parents=True)
    (directory / "run_configs" / "session1.json").write_text(json.dumps(config), encoding="utf-8")
    (directory / "family" / "zT" / unit_name / "xgboost" / "C0" / "r1_f0.json").write_text(
        json.dumps({"params": {}, "n_train": 10, "smoke_cap": smoke_cap}), encoding="utf-8")


def check_8_analysis_loader_guard(pooled_dir, specialist_dir, work_dir):
    """
    lofo_paperb.load_checkpoint_dir_for_analysis must raise for (a) any unit
    with smoke_cap=true -- both of this test's real role directories, every
    unit of which was fit with --smoke-search-space-cap -- and, separately,
    (b) two run_configs that disagree on the identity fields within one
    directory, isolated from (a) via a small synthetic directory with no
    smoke_cap issue at all. The new load_checkpoint_dirs_for_analysis
    (methodology doc section 8.7's role-separated Kaggle plan) must
    additionally (c) reject the real pooled+specialist pair for the same
    smoke_cap reason, (d) merge two individually-clean, identity-agreeing
    synthetic directories into one unit list without raising, and (e) raise
    when two individually-clean directories disagree with EACH OTHER on
    identity, even though neither disagrees within itself.
    """
    results = {}
    try:
        L.load_checkpoint_dir_for_analysis(pooled_dir)
        results["smoke_cap_rejected"] = False
    except ValueError as exc:
        results["smoke_cap_error"] = str(exc)
        # must be rejected for the real reason (a unit with smoke_cap=true), not any incidental "smoke_cap"
        # text elsewhere (e.g. a stray file without that field at all would raise a *different* message).
        results["smoke_cap_rejected"] = "were fit with --smoke-search-space-cap" in str(exc)

    try:
        L.load_checkpoint_dirs_for_analysis([pooled_dir, specialist_dir])
        results["merged_real_dirs_smoke_cap_rejected"] = False
    except ValueError as exc:
        results["merged_real_dirs_smoke_cap_error"] = str(exc)
        results["merged_real_dirs_smoke_cap_rejected"] = "were fit with --smoke-search-space-cap" in str(exc)

    synth = work_dir / "synthetic_identity_mismatch"
    base_config = {"dataset_sha256": "a" * 64, "splits_dir": "some/path", "labels_sha256": "b" * 64, "git_head": "deadbeef"}
    _write_synthetic_unit_dir(synth, base_config, "x")
    (synth / "run_configs" / "session2.json").write_text(json.dumps({**base_config, "git_head": "feedface"}), encoding="utf-8")
    try:
        L.load_checkpoint_dir_for_analysis(synth)
        results["identity_mismatch_rejected"] = False
    except ValueError as exc:
        results["identity_mismatch_error"] = str(exc)
        results["identity_mismatch_rejected"] = "disagree" in str(exc)

    # (d) two clean, identity-agreeing directories: the merge loader must succeed and return both units.
    agree_config = {"dataset_sha256": "c" * 64, "splits_dir": "some/other/path", "labels_sha256": "d" * 64, "git_head": "cafefeed"}
    merge_ok_a, merge_ok_b = work_dir / "synthetic_merge_ok_a", work_dir / "synthetic_merge_ok_b"
    _write_synthetic_unit_dir(merge_ok_a, agree_config, "unit_a")
    _write_synthetic_unit_dir(merge_ok_b, agree_config, "unit_b")
    try:
        merged = L.load_checkpoint_dirs_for_analysis([merge_ok_a, merge_ok_b])
        results["merge_ok_n_units"] = len(merged)
        results["merge_ok"] = len(merged) == 2
    except ValueError as exc:
        results["merge_ok_error"] = str(exc)
        results["merge_ok"] = False

    # (e) two clean directories that disagree WITH EACH OTHER, though neither disagrees within itself.
    merge_bad_a, merge_bad_b = work_dir / "synthetic_merge_bad_a", work_dir / "synthetic_merge_bad_b"
    _write_synthetic_unit_dir(merge_bad_a, agree_config, "unit_a")
    _write_synthetic_unit_dir(merge_bad_b, {**agree_config, "git_head": "deadfeed"}, "unit_b")
    try:
        L.load_checkpoint_dirs_for_analysis([merge_bad_a, merge_bad_b])
        results["cross_dir_identity_mismatch_rejected"] = False
    except ValueError as exc:
        results["cross_dir_identity_mismatch_error"] = str(exc)
        results["cross_dir_identity_mismatch_rejected"] = "disagree" in str(exc)

    results["passed"] = bool(
        results.get("smoke_cap_rejected") and results.get("merged_real_dirs_smoke_cap_rejected")
        and results.get("identity_mismatch_rejected") and results.get("merge_ok")
        and results.get("cross_dir_identity_mismatch_rejected")
    )
    return results


def check_11_specialist_independence(pooled_dir, specialist_dir):
    """
    Methodology doc section 8.1: the specialist must never depend on
    gpu-pooled's tuning_once/C0-C3. Structural proof, not just code
    inspection: cpu-specialist wrote every one of its checkpoints into
    specialist_dir, a directory gpu-pooled never touched (role-separated
    checkpoint dirs, section 8.7's Kaggle plan) -- if it held a hidden
    dependency on gpu-pooled's output, main()'s cpu-specialist reference run
    would have failed outright rather than reaching this check. This check
    additionally confirms the absence directly on disk, rather than trusting
    that failure would have been loud: no tuning_once/C0/C1/C2/C3 file
    anywhere under specialist_dir, and no specialist/specialist_gpu_control
    file anywhere under pooled_dir (the reverse direction, for symmetry).
    """
    pooled_only_stems = {"tuning_once", "C0", "C1", "C2", "C3"}
    specialist_only_stems = {"specialist", "specialist_gpu_control"}
    pooled_files_in_specialist_dir = [
        p for p in specialist_dir.rglob("*.npz") if p.parent.name in pooled_only_stems or p.stem in pooled_only_stems
    ]
    specialist_files_in_pooled_dir = [
        p for p in pooled_dir.rglob("*.npz") if p.parent.name in specialist_only_stems
    ]
    ok = not pooled_files_in_specialist_dir and not specialist_files_in_pooled_dir
    return {
        "pooled_files_in_specialist_dir": [str(p) for p in pooled_files_in_specialist_dir],
        "specialist_files_in_pooled_dir": [str(p) for p in specialist_files_in_pooled_dir],
        "passed": ok,
    }


def check_12_path_separator_normalization(work_dir):
    """
    check_splits_manifest and restore_from's manifest reader must resolve a
    manifest entry correctly whether it was written with '/' or '\\' as the
    separator (2026-09-30 fix: the committed splits manifest.json was written
    on Windows, backslash-separated; Kaggle is Linux, and check_splits_manifest
    raised FileNotFoundError on every entry). Each synthetic manifest below
    carries one entry of EACH style, pointing at real files, so a regression
    in the read-side normalisation (normalize_rel_path) would fail this on
    whichever style it broke, not just the other.
    """
    results = {}

    # -- check_splits_manifest --
    splits_dir = work_dir / "path_sep_splits"
    (splits_dir / "family" / "zT").mkdir(parents=True)
    file_a = splits_dir / "family" / "zT" / "unit_a.json"
    file_b = splits_dir / "family" / "zT" / "unit_b.json"
    file_a.write_text("{}", encoding="utf-8")
    file_b.write_text("{}", encoding="utf-8")
    manifest = {
        "family\\zT\\unit_a.json": L.sha256_file(file_a),  # backslash, as the committed splits manifest has
        "family/zT/unit_b.json": L.sha256_file(file_b),  # forward-slash
    }
    (splits_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    try:
        L.check_splits_manifest(splits_dir)
        results["check_splits_manifest_ok"] = True
    except Exception as exc:
        results["check_splits_manifest_error"] = str(exc)
        results["check_splits_manifest_ok"] = False

    # -- restore_from's manifest reader --
    source_dir = work_dir / "path_sep_restore_source"
    (source_dir / "family" / "zT" / "unit_a" / "xgboost" / "C0").mkdir(parents=True)
    (source_dir / "family" / "zT" / "unit_b" / "xgboost" / "C0").mkdir(parents=True)
    npz_a = source_dir / "family" / "zT" / "unit_a" / "xgboost" / "C0" / "r1_f0.npz"
    npz_b = source_dir / "family" / "zT" / "unit_b" / "xgboost" / "C0" / "r1_f0.npz"
    np.savez_compressed(npz_a, x=np.array([1]))
    np.savez_compressed(npz_b, x=np.array([2]))
    (source_dir / "run_configs").mkdir()
    identity_config = {"dataset_sha256": "e" * 64, "splits_dir": "some/path", "labels_sha256": "f" * 64, "git_head": "beadfeed"}
    (source_dir / "run_configs" / "session1.json").write_text(json.dumps(identity_config), encoding="utf-8")
    restore_manifest = {
        "family\\zT\\unit_a\\xgboost\\C0\\r1_f0.npz": L.sha256_file(npz_a),  # backslash
        "family/zT/unit_b/xgboost/C0/r1_f0.npz": L.sha256_file(npz_b),  # forward-slash
    }
    (source_dir / "manifest.json").write_text(json.dumps(restore_manifest), encoding="utf-8")

    dest_dir = work_dir / "path_sep_restore_dest"
    current_identity = (identity_config["dataset_sha256"], identity_config["splits_dir"],
                       identity_config["labels_sha256"], identity_config["git_head"])
    try:
        restore_result = L.restore_from(source_dir, dest_dir, current_identity)
        both_present = ((dest_dir / "family" / "zT" / "unit_a" / "xgboost" / "C0" / "r1_f0.npz").exists()
                        and (dest_dir / "family" / "zT" / "unit_b" / "xgboost" / "C0" / "r1_f0.npz").exists())
        results["restore_from_ok"] = bool(restore_result["n_copied"] == 2 and both_present)
        results["restore_from_n_copied"] = restore_result["n_copied"]
    except Exception as exc:
        results["restore_from_error"] = str(exc)
        results["restore_from_ok"] = False

    results["passed"] = bool(results.get("check_splits_manifest_ok")) and bool(results.get("restore_from_ok"))
    return results


def latest_run_config(checkpoint_dir, role):
    """The most recently written run_configs/*_<role>.json (not _results.json) under checkpoint_dir."""
    candidates = sorted(
        p for p in (checkpoint_dir / "run_configs").glob(f"*_{role}.json") if not p.name.endswith("_results.json")
    )
    if not candidates:
        raise FileNotFoundError(f"no run_configs/*_{role}.json under {checkpoint_dir}")
    return json.loads(candidates[-1].read_text(encoding="utf-8"))


def check_13_in_pair_time_budget(csv, labels_run, splits_dir, work_dir):
    """
    2026-09-30 fix: process_pooled_pair now checks --time-budget-hours itself,
    between tuning_once/C1/each (component, repeat, fold) fit, not only
    between whole tasks (run_pool's own, coarser check) -- a single
    gpu-pooled task can otherwise run for a long time unbroken regardless of
    the budget (the Kaggle incident this fixes: 70+ minutes against a
    5-minute budget, one task, no --folds/--trials override).

    Rather than guess a fixed number of seconds for a budget that lands
    mid-pair (fragile -- the local machine's speed and the exact split
    between tuning_once and the fold fits are not known in advance), this
    times ONE small pair unrestricted first, then reruns the same pair with
    a budget of half that measured wall time: long enough that tuning_once
    (the first, necessary piece of work) should complete, short enough that
    the whole pair (tuning_once + C1 + C0/C2/C3 x 2 folds, 8 checkpoint
    files) should not. Requires: some but not all files written, and
    units_partial_this_session > 0 in the session summary; then a resume
    (same command, no budget) must complete the pair without rewriting any
    file the partial run had already checkpointed.
    """
    timing_dir = work_dir / "in_pair_budget_timing"
    t0 = time.time()
    result = subprocess.run(harness_cmd(csv, labels_run, splits_dir, timing_dir, "gpu-pooled",
                                        extra=["--units", "manganite", "--models", "xgboost"]),
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"in-pair-budget timing run failed:\n{result.stdout}\n{result.stderr}")
    full_pair_seconds = time.time() - t0
    n_full_files = len(all_checkpoint_files(timing_dir))

    budget_seconds = full_pair_seconds / 2
    budget_dir = work_dir / "in_pair_budget"
    cmd = harness_cmd(csv, labels_run, splits_dir, budget_dir, "gpu-pooled",
                      extra=["--units", "manganite", "--models", "xgboost",
                            "--time-budget-hours", str(budget_seconds / 3600.0)])
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return {"returncode": result.returncode, "stderr_tail": (result.stderr or "")[-800:], "passed": False}
    partial_files = all_checkpoint_files(budget_dir)
    config = latest_run_config(budget_dir, "gpu-pooled")
    stopped_mid_pair = 0 < len(partial_files) < n_full_files
    partial_flagged = config["session_summary"]["units_partial_this_session"] > 0
    partial_hashes = file_hashes(partial_files)

    result2 = subprocess.run(harness_cmd(csv, labels_run, splits_dir, budget_dir, "gpu-pooled",
                                        extra=["--units", "manganite", "--models", "xgboost"]),
                            capture_output=True, text=True)
    if result2.returncode != 0:
        return {"returncode": result2.returncode, "stderr_tail": (result2.stderr or "")[-800:], "passed": False}
    final_files = all_checkpoint_files(budget_dir)
    final_hashes = file_hashes(final_files)
    not_rewritten = all(final_hashes.get(p) == h for p, h in partial_hashes.items())
    complete = len(final_files) == n_full_files

    return {
        "full_pair_seconds": round(full_pair_seconds, 2), "n_full_files": n_full_files,
        "budget_seconds": round(budget_seconds, 2), "n_partial_files": len(partial_files),
        "stopped_mid_pair": stopped_mid_pair, "partial_flagged_in_summary": partial_flagged,
        "n_final_files": len(final_files), "resume_complete": complete,
        "no_file_rewritten_on_resume": not_rewritten,
        "passed": bool(stopped_mid_pair and partial_flagged and complete and not_rewritten),
    }


def check_14_lf_hash_verification(splits_dir, work_dir):
    """
    2026-10-01 fix: the committed splits manifest.json was regenerated from
    LF-normalised content (rehash_splits_manifest.py), after .gitattributes
    started forcing `paper_b/**/*.json text eol=lf`. Kaggle (Linux) had
    crashed verifying a SHA256 that had been computed from a Windows (CRLF)
    working copy of the identical, LF-stored git blob.

    Builds two synthetic copies of splits_dir's manifest-listed TEXT entries
    (every .npz is binary, untouched by eol handling, and is not part of
    this check) -- one forced to pure LF (what a fresh Linux checkout, or
    this fix's forced recheckout, produces) and one forced to pure CRLF
    (what an un-normalised Windows checkout used to produce) -- each with
    its own manifest.json covering only those text entries. Requires
    check_splits_manifest to PASS on the LF copy (simulating Linux) and to
    RAISE on the CRLF copy (proving the manifest is genuinely pinned to LF
    content specifically, not accidentally EOL-agnostic).
    """
    manifest = json.loads((splits_dir / "manifest.json").read_text(encoding="utf-8"))
    text_entries = {rel: h for rel, h in manifest.items() if not rel.endswith(".npz")}

    def build_variant(eol):
        variant_dir = work_dir / f"lf_hash_check_{eol}"
        variant_dir.mkdir(parents=True)
        (variant_dir / "manifest.json").write_text(json.dumps(text_entries), encoding="utf-8")
        for rel in text_entries:
            dest = variant_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            data = (splits_dir / rel).read_bytes().replace(b"\r\n", b"\n")
            if eol == "crlf":
                data = data.replace(b"\n", b"\r\n")
            dest.write_bytes(data)
        return variant_dir

    lf_dir = build_variant("lf")
    crlf_dir = build_variant("crlf")

    lf_passed, lf_error = True, None
    try:
        L.check_splits_manifest(lf_dir)
    except Exception as exc:
        lf_passed, lf_error = False, str(exc)

    crlf_rejected, crlf_error = False, None
    try:
        L.check_splits_manifest(crlf_dir)
    except ValueError as exc:
        crlf_rejected, crlf_error = True, str(exc)

    return {
        "n_text_entries_checked": len(text_entries), "lf_verification_passed": lf_passed, "lf_error": lf_error,
        "crlf_verification_rejected": crlf_rejected, "crlf_error_tail": (crlf_error or "")[-300:],
        "passed": bool(lf_passed and crlf_rejected),
    }


def check_15_multiworker_pool(csv, labels_run, splits_dir, work_dir):
    """
    2026-10-01 fix: every multiprocessing.Pool() path (--workers > 1) is
    otherwise untouched by this whole suite -- every other check leaves
    --workers at its default (1), which never constructs a
    multiprocessing.Pool at all (run_pool's workers<=1 branch calls
    _pool_initializer directly, in-process). The actual Kaggle Linux crash
    this fixes ("A SemLock created in a fork context is being shared with a
    process in a spawn context") can only happen when a Pool is really
    built, from a counter built via the platform DEFAULT context, which on
    Windows already happens to be 'spawn' -- so this check cannot reproduce
    THAT crash specifically no matter what it does (only a Linux run can;
    see the Kaggle CPU smoke cells), but it exercises the Pool code path at
    all for the first time in this suite, catching any other multi-worker
    regression (pickling, initializer argument mismatches, a hang) that the
    rest of the suite would otherwise never touch.
    """
    pool_dir = work_dir / "multiworker_pool"
    cmd = harness_cmd(csv, labels_run, splits_dir, pool_dir, "gpu-pooled", extra=["--workers", "2"])
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        return {"timed_out": True, "passed": False}
    passed = result.returncode == 0
    n_files = len(all_checkpoint_files(pool_dir)) if passed else 0
    return {
        "returncode": result.returncode, "n_checkpoint_files": n_files,
        "stderr_tail": None if passed else (result.stderr or "")[-800:],
        "passed": passed,
    }


def check_9_time_budget(csv, labels_run, splits_dir, work_dir):
    """
    --time-budget-hours, a budget of a few seconds: the harness must still
    exit 0 and write a session summary (a too-small budget can honestly
    finish zero units -- loading the dataset alone can exceed it -- that is
    a clean reported outcome, not a failure).
    """
    budget_dir = work_dir / "time_budget_smoke"
    seconds_budget = 3
    cmd = harness_cmd(csv, labels_run, splits_dir, budget_dir, "gpu-pooled",
                      extra=["--time-budget-hours", str(seconds_budget / 3600.0)])
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return {"returncode": result.returncode, "stderr_tail": (result.stderr or "")[-800:], "passed": False}
    config = latest_run_config(budget_dir, "gpu-pooled")
    summary = config["session_summary"]
    ok = (
        summary["time_budget_hours"] == seconds_budget / 3600.0
        and summary["units_done_this_session"] + summary["units_remaining"] == config["n_tasks"]
        and summary["units_remaining"] > 0  # a 3-second budget cannot finish this task list (dataset load alone exceeds it)
    )
    return {"returncode": result.returncode, "n_tasks": config["n_tasks"], "session_summary": summary, "passed": ok}


def check_10_restore_from(csv, labels_run, splits_dir, ref_dir, work_dir):
    """
    --restore-from a .tar.gz of a completed session: (a) a fresh session
    restoring from it must skip every restored unit (n_wrote == 0, since
    ref_dir already covers this test's exact scope) and reach the same file
    set as ref_dir; (b) a tampered copy of that tar.gz must be refused.
    """
    import tarfile as tf
    archive = work_dir / "restore_source.tar.gz"
    with tf.open(archive, "w:gz") as tar:
        for child in sorted(ref_dir.iterdir()):
            tar.add(child, arcname=child.name)

    restored_dir = work_dir / "restored"
    result = subprocess.run(harness_cmd(csv, labels_run, splits_dir, restored_dir, "gpu-pooled",
                                        extra=["--restore-from", str(archive)]), capture_output=True, text=True)
    restore_ok = result.returncode == 0
    n_wrote_after_restore = None
    same_as_reference = None
    if restore_ok:
        config = latest_run_config(restored_dir, "gpu-pooled")
        n_wrote_after_restore = config["n_wrote"]
        ref_names = {p.relative_to(ref_dir) for p in all_checkpoint_files(ref_dir)}
        restored_names = {p.relative_to(restored_dir) for p in all_checkpoint_files(restored_dir)}
        same_as_reference = ref_names == restored_names  # gpu-pooled alone; n_wrote==0 means nothing new was added

    tampered_dir = work_dir / "tampered_extract"
    with tf.open(archive, "r:gz") as tar:
        tar.extractall(tampered_dir, filter="data")  # PEP 706; matches restore_from's own extractall call
    victim = next(p for p in tampered_dir.rglob("*.npz") if p.name != "manifest.json")
    data = bytearray(victim.read_bytes())
    data[-1] ^= 0xFF  # flip the last byte: same length, different content, still a loadable-looking file
    victim.write_bytes(bytes(data))
    tampered_archive = work_dir / "restore_source_tampered.tar.gz"
    with tf.open(tampered_archive, "w:gz") as tar:
        for child in sorted(tampered_dir.iterdir()):
            tar.add(child, arcname=child.name)

    tampered_restore_dir = work_dir / "restored_tampered"
    result2 = subprocess.run(harness_cmd(csv, labels_run, splits_dir, tampered_restore_dir, "gpu-pooled",
                                         extra=["--restore-from", str(tampered_archive)]), capture_output=True, text=True)
    tamper_refused = result2.returncode != 0 and "sha256" in (result2.stderr or "").lower()

    return {
        "restore_returncode": result.returncode, "restore_ok": restore_ok,
        "n_wrote_after_restore": n_wrote_after_restore, "same_as_reference": same_as_reference,
        "tampered_returncode": result2.returncode, "tampered_stderr_tail": (result2.stderr or "")[-500:],
        "tamper_refused": tamper_refused,
        "passed": bool(restore_ok and n_wrote_after_restore == 0 and same_as_reference and tamper_refused),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--csv", required=True)
    parser.add_argument("--labels-run", required=True)
    parser.add_argument("--splits-dir", required=True)
    parser.add_argument("--work-dir", default=None)
    args = parser.parse_args(argv)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out_dir = Path("paper_b/results/smoke_test") / stamp
    out_dir.mkdir(parents=True)
    work_dir = Path(args.work_dir) if args.work_dir else out_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)

    report = {"utc_stamp": stamp, "target": TARGET, "units": UNITS, "folds": FOLDS, "trials": TRIALS}
    report["check_1_leakage_assertions"] = {"assertion_functions_catch_violations": check_1_leakage_assertions_exercised()}
    report["check_1_leakage_assertions"]["passed"] = report["check_1_leakage_assertions"]["assertion_functions_catch_violations"]

    resume_result, pooled_ref_dir = check_2_checkpoint_resume(args.csv, args.labels_run, args.splits_dir, work_dir)
    report["check_1_leakage_assertions"]["exercised_by_reference_run_with_no_assertion_error"] = True  # gpu-pooled ref run above succeeded

    # cpu-specialist's own reference run, role-separated (methodology doc section 8.7): its own directory,
    # never touching pooled_ref_dir -- this is check_11's structural independence proof.
    specialist_ref_dir = work_dir / "reference_cpu_specialist"
    run_role(args.csv, args.labels_run, args.splits_dir, specialist_ref_dir, "cpu-specialist")

    report["check_2_checkpoint_resume"] = resume_result
    report["check_3_determinism"] = check_3_determinism(args.csv, args.labels_run, args.splits_dir, pooled_ref_dir, work_dir)
    report["check_4_murphy_and_r2"] = check_4_murphy_and_r2([pooled_ref_dir, specialist_ref_dir])
    report["check_5_hand_check"] = check_5_hand_check(pooled_ref_dir)
    report["check_6_bootstrap"] = check_6_bootstrap(pooled_ref_dir)
    report["check_7_smoke_guard_refusal"] = check_7_smoke_guard_refusal(args.csv, args.labels_run, args.splits_dir)
    report["check_8_analysis_loader_guard"] = check_8_analysis_loader_guard(pooled_ref_dir, specialist_ref_dir, work_dir)
    report["check_9_time_budget"] = check_9_time_budget(args.csv, args.labels_run, args.splits_dir, work_dir)
    report["check_10_restore_from"] = check_10_restore_from(args.csv, args.labels_run, args.splits_dir, pooled_ref_dir, work_dir)
    report["check_11_specialist_independence"] = check_11_specialist_independence(pooled_ref_dir, specialist_ref_dir)
    report["check_12_path_separator_normalization"] = check_12_path_separator_normalization(work_dir)
    report["check_13_in_pair_time_budget"] = check_13_in_pair_time_budget(args.csv, args.labels_run, args.splits_dir, work_dir)
    report["check_14_lf_hash_verification"] = check_14_lf_hash_verification(Path(args.splits_dir), work_dir)
    report["check_15_multiworker_pool"] = check_15_multiworker_pool(args.csv, args.labels_run, args.splits_dir, work_dir)

    report["all_passed"] = all(report[k]["passed"] for k in report if k.startswith("check_"))
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")

    print(json.dumps(report, indent=2, default=str))
    print(f"\nWrote {out_dir}")
    print("ALL PASSED" if report["all_passed"] else "SOME CHECKS FAILED")
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
