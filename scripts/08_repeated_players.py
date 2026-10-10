"""Checks for repeated players: grouped cross-validation and player-clustered standard errors.

The same players appear in multiple seasons (3,532 rows from 888 players), so rows are not
independent. Two checks:

1. Predictive accuracy. Repeated 5-fold CV with folds assigned by row versus folds assigned by
   player (every season of a player lands in the same fold, so the model is always tested on
   players it has never seen). Full 500+ minute sample, plain OLS (the elastic net gives nearly
   identical coefficients). Fold assignment uses a small explicit linear congruential generator
   so the folds are identical on every machine and library version.

2. Inference. Full-sample standardized OLS with classical standard errors and with
   cluster-robust (CR1) standard errors clustered by player; p-values for the clustered
   version use a t distribution with (number of players - 1) degrees of freedom.

Output:
  results/repeated_players.json
"""
import json

import numpy as np
from scipy import stats

from common import MODEL_STATS, RESULTS, SEED, TARGET, ensure_dirs, load_model_data

N_SPLITS = 5
N_REPEATS = 20


def permutation(n, seed):
    """Fisher-Yates shuffle driven by a 32-bit LCG (Numerical Recipes constants)."""
    state = seed & 0xFFFFFFFF
    order = list(range(n))
    for i in range(n - 1, 0, -1):
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        j = state % (i + 1)
        order[i], order[j] = order[j], order[i]
    return order


def fit_predict(X_tr, y_tr, X_te):
    A = np.column_stack([np.ones(len(X_tr)), X_tr])
    beta, *_ = np.linalg.lstsq(A, y_tr, rcond=None)
    return np.column_stack([np.ones(len(X_te)), X_te]) @ beta


def repeated_cv(X, y, unit_of_row, units):
    r2s, rmses = [], []
    for r in range(N_REPEATS):
        order = permutation(len(units), SEED * 1000 + r)
        fold_of_unit = {units[u]: pos % N_SPLITS for pos, u in enumerate(order)}
        folds = np.array([fold_of_unit[u] for u in unit_of_row])
        for f in range(N_SPLITS):
            te = folds == f
            resid = y[te] - fit_predict(X[~te], y[~te], X[te])
            r2s.append(1 - (resid ** 2).sum() / ((y[te] - y[te].mean()) ** 2).sum())
            rmses.append(np.sqrt((resid ** 2).mean()))
    r2s = np.array(r2s)
    return {"r2_mean": round(float(r2s.mean()), 4), "r2_sd": round(float(r2s.std(ddof=1)), 4),
            "r2_min": round(float(r2s.min()), 4), "r2_max": round(float(r2s.max()), 4),
            "rmse_mean": round(float(np.mean(rmses)), 4)}


def fmt_p(p):
    return "<.0001" if p < 0.0001 else f"{p:.4f}"


def clustered_ols(X, y, groups):
    """Standardized OLS with classical and player-clustered (CR1) standard errors."""
    Z = (X - X.mean(0)) / X.std(0, ddof=1)
    yz = (y - y.mean()) / y.std(ddof=1)
    A = np.column_stack([np.ones(len(Z)), Z])
    n, k = A.shape
    bread = np.linalg.inv(A.T @ A)
    beta = bread @ A.T @ yz
    e = yz - A @ beta
    se_classical = np.sqrt(np.diag(bread) * (e @ e) / (n - k))

    labels = np.asarray(groups)
    uniq = np.unique(labels)
    G = len(uniq)
    meat = np.zeros((k, k))
    for g in uniq:
        s = A[labels == g].T @ e[labels == g]
        meat += np.outer(s, s)
    V = bread @ meat @ bread * (G / (G - 1)) * ((n - 1) / (n - k))
    se_clustered = np.sqrt(np.diag(V))

    rows = []
    for j, name in enumerate(MODEL_STATS, start=1):
        t_cl = beta[j] / se_clustered[j]
        p_cl = 2 * stats.t.sf(abs(t_cl), G - 1)
        p_classical = 2 * stats.t.sf(abs(beta[j] / se_classical[j]), n - k)
        rows.append({"stat": name, "beta": round(float(beta[j]), 4),
                     "se_classical": round(float(se_classical[j]), 4), "p_classical": fmt_p(p_classical),
                     "se_clustered": round(float(se_clustered[j]), 4), "t_clustered": round(float(t_cl), 3),
                     "p_clustered": fmt_p(p_cl)})
    return G, rows


def main():
    ensure_dirs()
    d = load_model_data()
    X, y = d[MODEL_STATS].values.astype(float), d[TARGET].values.astype(float)

    rows = list(range(len(d)))
    players = sorted(d["PlayerID"].unique())
    cv = {"n_splits": N_SPLITS, "n_repeats": N_REPEATS, "model": "OLS on the 14 model stats",
          "row_folds": repeated_cv(X, y, rows, rows),
          "player_grouped_folds": repeated_cv(X, y, d["PlayerID"].tolist(), players)}
    G, coefs = clustered_ols(X, y, d["PlayerID"].values)

    out = {"n_rows": len(d), "n_players": G, "cross_validation": cv, "clustered_ols": coefs}
    (RESULTS / "repeated_players.json").write_text(json.dumps(out, indent=2) + "\n")

    print(f"Row-level folds:      CV R2 {cv['row_folds']['r2_mean']:.3f} (SD {cv['row_folds']['r2_sd']:.3f})")
    print(f"Player-grouped folds: CV R2 {cv['player_grouped_folds']['r2_mean']:.3f} "
          f"(SD {cv['player_grouped_folds']['r2_sd']:.3f})")
    for r in coefs:
        print(f"{r['stat']:>5}  beta {r['beta']:+.3f}  p classical {r['p_classical']:>7}  p clustered {r['p_clustered']:>7}")


if __name__ == "__main__":
    main()
