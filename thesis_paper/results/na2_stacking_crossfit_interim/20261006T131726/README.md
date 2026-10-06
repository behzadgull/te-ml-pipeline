# NA2 stacking, interim cross-fitted version (NOT the pre-stated design)

`scripts/kaggle/na2_stacking.py` (non-negative ridge, alpha 1, cross-fitted across the five outer folds of a repeat on the base models' outer out-of-fold predictions). The base predictions of the
meta-training folds come from models trained on rows that include the held-out fold, so the stack is not fully nested. The design stated for the paper (meta-learner trained on inner
chemistry-cluster out-of-fold predictions inside each outer training fold, never in-fold) is `scripts/kaggle/na2_stacking_nested.py`. **Nothing from this folder is quoted in `paper.md`.**
The simple mean of the three base models (`mean3` in `stacking_results.json`) involves no fitted meta-learner and is not affected by that leak.
