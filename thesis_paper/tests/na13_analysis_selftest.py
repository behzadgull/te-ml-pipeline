"""
Self-test of scripts/na13_analysis.py on SYNTHETIC input. Every number it prints or writes is synthetic and is NOT a result; nothing it creates is used by the paper.

`make` builds four fake NA13 bundles (one per target, 25 units each) in a scratch folder: the real folds and chemistry clusters are rebuilt from the snapfix CSV, so the analysis' fold check passes, but the predictions are the
true values plus random noise (the all-397 "fit" less noisy than the "selected" fit, so the paired difference is negative by construction) and the bundle metadata is written to satisfy the pre-registered conditions
(pinned commit, cuda, tree_clean, complete, 25 units, manifest). `test` runs the analysis on them (positive case, rerun determinism) and then on mutated copies that each break one of the refusal conditions
(missing or extra unit, a file other than the top-level README.md outside the manifest, other commit, device cpu, smoke, dirty tree, incomplete status, wrong dataset, lasso_max_iter 2000, predictions altered after the manifest, three bundles, a target twice, n_boot other than 2000
without --out-dir). `all` (the default) does both and then removes the scratch folder.

Usage (repository root; needs data/processed/featurized_ThermoelectricMaterials_2026-08-22-snapfix.csv; about ten minutes):
    python thesis_paper/tests/na13_analysis_selftest.py [make|test|all] [scratch folder]
"""
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / 'thesis_paper/scripts')); sys.path.insert(0, str(REPO / 'thesis_paper/scripts/kaggle'))
from src import nested_cv as ncv  # noqa: E402
import na13_analysis as A  # noqa: E402

PY = sys.executable
MODE = sys.argv[1] if len(sys.argv) > 1 else 'all'
ROOT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(tempfile.gettempdir()) / 'na13_synth'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_manifest(d):
    files = {f.relative_to(d).as_posix(): sha(f) for f in sorted(d.rglob('*')) if f.is_file() and f.name != 'manifest.json' and f.relative_to(d).as_posix() != 'README.md'}  # a top-level README.md lies outside the manifest, as in the committed bundles
    (d / 'manifest.json').write_text(json.dumps({'files': files}), encoding='utf-8')


def make(target, df_head, df):
    d = ROOT / f'na13_{target}'
    shutil.rmtree(d, ignore_errors=True)
    (d / 'units').mkdir(parents=True); (d / 'run_configs').mkdir()
    sub = df[df[target].notna()].reset_index(drop=True)
    y = ncv._transform_target(sub[target].to_numpy(float), target)
    groups = sub[ncv.GROUP_COL].to_numpy()
    cols = ncv.get_feature_columns(df_head)
    rng_master = np.random.default_rng(0)
    gen = np.random.default_rng(123 + len(target))
    for r in range(5):
        rng = np.random.default_rng(rng_master.integers(0, 2**32 - 1))
        for f, (tr, te) in enumerate(ncv.outer_splits('chemistry', len(sub), {'chemistry': groups}, 5, rng)):
            yt = y[te]; sd = yt.std()
            p_all = yt + gen.normal(0, 0.5 * sd, len(yt))
            p_sel = yt + gen.normal(0, 0.55 * sd, len(yt))
            r2 = lambda p: 1 - np.sum((yt - p) ** 2) / np.sum((yt - yt.mean()) ** 2)  # noqa: E731
            sel = list(gen.choice(cols, A.K_THESIS[target], replace=False))
            grid = [float(10 ** -i) for i in range(20)]
            idx = int(gen.integers(14, 20))
            meta = {'target': target, 'repeat': r, 'fold': f, 'n_train': int(len(tr)), 'n_test': int(len(te)), 'n_after_pearson': 250, 'n_after_lasso': int(gen.integers(140, 200)), 'lasso_alpha': grid[idx], 'lasso_alpha_index': idx,
                    'lasso_alphas': grid, 'n_convergence_warnings': 0, 'n_convergence_warnings_inner_paths': 0, 'final_refit_converged': True, 'final_refit_dual_gap': 0.1, 'final_refit_tolerance': 1.0, 'final_refit_n_iter': 100,
                    'lasso_max_iter': 20000, 'device': 'cuda', 'n_selected': A.K_THESIS[target], 'selected': sel, 'outer_r2': r2(p_sel), 'r2_selected': r2(p_sel), 'r2_all397': r2(p_all),
                    'delta_r2_selected_minus_all397': r2(p_sel) - r2(p_all), 'seconds_selected_fit': 1.0, 'seconds_all397_fit': 2.0, 'seconds': 3.0}
            uid = f'{target}_repeat{r}_fold{f}'
            (d / 'units' / f'{uid}.json').write_text(json.dumps({'unit': uid, 'meta': meta, 'has_arrays': True}), encoding='utf-8')
            np.savez_compressed(d / 'units' / f'{uid}.npz', y_true=yt, y_pred=p_sel, y_pred_all397=p_all)
    cfg = {'git_head': A.PIN, 'tree_clean': True, 'allow_dirty': False, 'smoke': False, 'dataset_sha256': A.DATASET_SHA256, 'device': 'cuda',
           'params': {'targets': [target], 'n_repeats': 5, 'n_folds': 5, 'seed': 0, 'smoke': False, 'k': A.K_THESIS, 'lasso_max_iter': 20000, 'device': 'cuda', 'paired_all397': True}}
    (d / 'run_configs' / 'session_01.json').write_text(json.dumps(cfg), encoding='utf-8')
    (d / 'status.json').write_text(json.dumps({'complete': True, 'units_total': 25, 'units_done': 25, 'accepted_as_result': True}), encoding='utf-8')
    (d / 'results.json').write_text('{}', encoding='utf-8')
    write_manifest(d)
    (d / 'README.md').write_text('synthetic bundle README (outside the manifest, as in the committed bundles)', encoding='utf-8')
    return d


def run(bundles, extra=(), out='out'):
    r = subprocess.run([PY, str(REPO / 'thesis_paper/scripts/na13_analysis.py'), '--bundles', ','.join(map(str, bundles)), '--out-dir', str(ROOT / out), *extra], cwd=REPO, capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr).strip().splitlines()


def mutated(src, name, fn):
    d = ROOT / name
    shutil.rmtree(d, ignore_errors=True)
    shutil.copytree(src, d)
    fn(d)
    write_manifest(d)
    return d


def mutated_after_manifest(src, name, fn):
    d = ROOT / name
    shutil.rmtree(d, ignore_errors=True)
    shutil.copytree(src, d)
    fn(d)  # the manifest is NOT rewritten
    return d


def edit_cfg(d, **kw):
    p = d / 'run_configs' / 'session_01.json'
    c = json.loads(p.read_text(encoding='utf-8'))
    for k, v in kw.items():
        if k.startswith('params.'):
            c['params'][k[7:]] = v
        else:
            c[k] = v
    p.write_text(json.dumps(c), encoding='utf-8')


def do_make():
    ROOT.mkdir(exist_ok=True)
    head = pd.read_csv(REPO / A.DATASET, nrows=0)
    df = pd.read_csv(REPO / A.DATASET)
    for t in A.TARGETS:
        make(t, head, df)
        print('made synthetic bundle', t, flush=True)


def do_test():
    good = [ROOT / f'na13_{t}' for t in A.TARGETS]
    shutil.rmtree(ROOT / 'out', ignore_errors=True)
    code, out = run(good)
    print('POSITIVE (SYNTHETIC input, n_boot 2000): exit', code, '(expected 0)'); print('\n'.join(out[-6:]))
    d = ROOT / 'out'
    print('files:', sorted(p.name for p in d.iterdir()))
    a = json.loads((d / 'analysis.json').read_text(encoding='utf-8'))
    for t in A.TARGETS:
        pdiff = a['targets'][t]['paired_difference_selected_minus_all397']
        print(f"  {t}: paired difference {pdiff['mean']:+.4f}, interval {[round(x, 4) for x in pdiff['ci95_cluster_bootstrap']]}, {pdiff['verdict']}  (SYNTHETIC)")
    print('per_fold rows', sum(1 for _ in open(d / 'per_fold.csv')) - 1, '(expected 100) | selection_frequency rows', sum(1 for _ in open(d / 'selection_frequency.csv')) - 1)
    rc = json.loads((d / 'run_config.json').read_text(encoding='utf-8'))
    print('run_config inputs:', sorted(rc['inputs']))
    code2, _ = run(good, out='out2')
    a2 = json.loads((ROOT / 'out2/analysis.json').read_text(encoding='utf-8'))
    same = all(a['targets'][x]['paired_difference_selected_minus_all397']['ci95_cluster_bootstrap'] == a2['targets'][x]['paired_difference_selected_minus_all397']['ci95_cluster_bootstrap'] for x in A.TARGETS)
    print('rerun gives identical intervals:', same and code2 == 0)
    failures = []
    S = ROOT / 'na13_kappa'

    def drop_unit(d):
        for ext in ('json', 'npz'):
            (d / 'units' / f'kappa_repeat4_fold4.{ext}').unlink()

    def extra_unit(d):
        shutil.copy(d / 'units' / 'kappa_repeat0_fold0.json', d / 'units' / 'kappa_repeat9_fold0.json')

    def incomplete(d):
        (d / 'status.json').write_text(json.dumps({'complete': False, 'units_total': 25, 'units_done': 24, 'accepted_as_result': False}), encoding='utf-8')

    tests = {
        'a unit missing (manifest rewritten)': mutated(S, 't1', drop_unit),
        'an unexpected extra unit': mutated(S, 't2', extra_unit),
        'a different code commit': mutated(S, 't3', lambda d: edit_cfg(d, git_head='47339cfafaa3efc53caafa69133710bb466a4782')),
        'device cpu': mutated(S, 't4', lambda d: edit_cfg(d, device='cpu')),
        'a smoke run': mutated(S, 't5', lambda d: edit_cfg(d, smoke=True)),
        'tree_clean false': mutated(S, 't6', lambda d: edit_cfg(d, tree_clean=False)),
        'status incomplete': mutated(S, 't7', incomplete),
        'wrong dataset sha': mutated(S, 't8', lambda d: edit_cfg(d, dataset_sha256='0' * 64)),
        'lasso_max_iter 2000': mutated(S, 't9', lambda d: edit_cfg(d, **{'params.lasso_max_iter': 2000})),
        'a file other than README.md outside the manifest': mutated_after_manifest(S, 't11', lambda d: (d / 'notes.txt').write_text('x', encoding='utf-8')),
        'a README.md inside units/ (only the top-level README.md is allowed)': mutated_after_manifest(S, 't12', lambda d: (d / 'units' / 'README.md').write_text('x', encoding='utf-8')),
        'predictions altered after the manifest (no rewrite)': None,
    }
    for name, d in tests.items():
        if d is None:
            d = ROOT / 't10'; shutil.rmtree(d, ignore_errors=True); shutil.copytree(S, d)
            np.savez_compressed(d / 'units' / 'kappa_repeat0_fold0.npz', y_true=np.zeros(3), y_pred=np.zeros(3), y_pred_all397=np.zeros(3))
        code, out = run([d if x.name == 'na13_kappa' else x for x in good])
        ok = code == 1 and 'REFUSED' in (out[-1] if out else '')
        failures += [] if ok else [name]
        print(f'REFUSAL TEST {name}: exit {code} {"OK" if ok else "FAILED"} | {out[-1][:200] if out else ""}')
    for name, bundles in (('only three bundles', good[:3]), ('two bundles of the same target', [good[0], good[0], good[2], good[3]])):
        code, out = run(bundles)
        ok = code == 1 and 'REFUSED' in out[-1]
        failures += [] if ok else [name]
        print(f'REFUSAL TEST {name}: exit {code} {"OK" if ok else "FAILED"} | {out[-1][:160]}')
    code, _ = run(good, ['--n-boot', '50'], out='out3')
    print('n_boot 50 with --out-dir (allowed for tests): exit', code, '(expected 0)')
    r = subprocess.run([PY, str(REPO / 'thesis_paper/scripts/na13_analysis.py'), '--bundles', ','.join(map(str, good)), '--n-boot', '50'], cwd=REPO, capture_output=True, text=True)
    ok = r.returncode == 1 and 'REFUSED' in (r.stdout + r.stderr)
    failures += [] if ok else ['n_boot 50 without --out-dir']
    print('REFUSAL TEST n_boot 50 without --out-dir: exit', r.returncode, 'OK' if ok else 'FAILED')
    print('ALL TESTS PASSED (synthetic input; no number above is a result)' if not failures and code == 0 and same else f'FAILED: {failures}')


if MODE in ('make', 'all'):
    do_make()
if MODE in ('test', 'all'):
    do_test()
if MODE == 'all':
    shutil.rmtree(ROOT, ignore_errors=True)
