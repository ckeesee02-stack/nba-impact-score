"""Relative weights analysis (Johnson, 2000) with bootstrap significance testing
(Tonidandel, LeBreton & Johnson, 2009): each stat's weight is compared with a random noise variable
across 5,000 bootstrap resamples.

Output: results/relative_weights.csv
"""
import numpy as np
import pandas as pd

from common import MODEL_STATS, RESULTS, SEED, TARGET, ensure_dirs, load_model_data, relative_weights

N_BOOT = 5000


def main():
    ensure_dirs()
    d = load_model_data()
    X, y = d[MODEL_STATS].values.astype(float), d[TARGET].values.astype(float)
    n, p = X.shape
    w = relative_weights(X, y)

    rng = np.random.default_rng(SEED)
    diffs, boot = np.zeros((N_BOOT, p)), np.zeros((N_BOOT, p))
    for i in range(N_BOOT):
        idx = rng.integers(0, n, n)
        ww = relative_weights(np.column_stack([X[idx], rng.standard_normal(n)]), y[idx])
        boot[i], diffs[i] = ww[:p], ww[:p] - ww[p]

    lo, hi = np.percentile(diffs, [2.5, 97.5], axis=0)
    ci = np.percentile(boot, [2.5, 97.5], axis=0)
    p_boot = np.minimum(1, 2 * np.minimum((diffs <= 0).mean(0), (diffs >= 0).mean(0)))
    r = [np.corrcoef(X[:, j], y)[0, 1] for j in range(p)]

    out = pd.DataFrame({
        "stat": MODEL_STATS, "raw_weight": w, "pct_of_r2": 100 * w / w.sum(),
        "ci_low": ci[0], "ci_high": ci[1], "p_vs_random": p_boot,
        "significant_vs_random": lo > 0, "raw_correlation": r,
    }).sort_values("raw_weight", ascending=False)
    out.to_csv(RESULTS / "relative_weights.csv", index=False)
    print(f"Sum of raw weights (= OLS R2): {w.sum():.4f}")
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
