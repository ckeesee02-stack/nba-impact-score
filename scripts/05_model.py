"""Model: elastic net (tuned by 5-fold CV on the training set), 80/20 evaluation,
repeated 5-fold cross-validation, and full-sample OLS with classical and HC3 standard errors.

Outputs:
  results/elastic_net_tuning.csv   CV grid results
  results/model_metrics.json       train/test/CV metrics, OLS fit, F-test
  results/elastic_net_coefs.csv    standardized elastic net coefficients (training fit)
  results/ols_coefficients.csv     standardized OLS betas, SEs, t, p (classical and HC3)
  results/ols_summary.txt          R-style summary(lm) printout
  results/cutoff_sensitivity.csv   cross-validated R2 at alternative minutes cutoffs
"""
import json

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV, KFold, RepeatedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import (EN_ALPHA, EN_L1_RATIO, MODEL_STATS, RESULTS, SEED, TARGET,
                    add_impact_score, ensure_dirs, load_model_data, load_player_seasons)


def en_pipeline(alpha=EN_ALPHA, l1=EN_L1_RATIO):
    return make_pipeline(StandardScaler(), ElasticNet(alpha=alpha, l1_ratio=l1, max_iter=100_000, tol=1e-10))


def r2(y, p):
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def rmse(y, p):
    return float(np.sqrt(((y - p) ** 2).mean()))


def ols(X, y):
    """OLS on z-scored X and y. Returns betas, classical SE, HC3 SE, residuals."""
    Z = (X - X.mean(0)) / X.std(0, ddof=1)
    yz = (y - y.mean()) / y.std(ddof=1)
    A = np.column_stack([np.ones(len(yz)), Z])
    XtXi = np.linalg.inv(A.T @ A)
    b = XtXi @ A.T @ yz
    e = yz - A @ b
    dfr = len(yz) - A.shape[1]
    se = np.sqrt(np.diag(XtXi) * (e @ e) / dfr)
    h = np.einsum("ij,jk,ik->i", A, XtXi, A)
    meat = (A * (e ** 2 / (1 - h) ** 2)[:, None]).T @ A
    se_hc3 = np.sqrt(np.diag(XtXi @ meat @ XtXi))
    return b, se, se_hc3, e, dfr


def r_style_summary(names, b, se, t, p, e, dfr, r2_, adj, F, k):
    def star(v): return "***" if v < .001 else "**" if v < .01 else "*" if v < .05 else "." if v < .1 else ""
    def pf(v):
        if v < 2e-16: return "< 2e-16"
        if v < 1e-4: return f"{v:.2e}"
        return f"{v:.6f}" if v < .001 else f"{v:.4f}"
    def bt(v): return v if v.isidentifier() or v == "(Intercept)" else f"`{v}`"
    q = np.percentile(e, [0, 25, 50, 75, 100])
    terms = " + ".join(bt(n) for n in names[1:])
    L = ["Call:", f"lm(formula = ImpactScore ~ {terms}, data = model_data_standardized)", "",
         "Residuals:", "      Min        1Q    Median        3Q       Max",
         " ".join(f"{v:9.4f}" for v in q), "", "Coefficients:",
         f"{'':12}{'Estimate':>10} {'Std. Error':>10} {'t value':>8} {'Pr(>|t|)':>9}"]
    for nm, bb, ss, tt, pp in zip(names, b, se, t, p):
        L.append(f"{bt(nm):12}{bb:10.6f} {ss:10.6f} {tt:8.3f} {pf(pp):>9} {star(pp)}")
    L += ["---", "Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1", "",
          f"Residual standard error: {np.sqrt(e @ e / dfr):.4f} on {dfr} degrees of freedom",
          f"Multiple R-squared:  {r2_:.4f},    Adjusted R-squared:  {adj:.4f}",
          f"F-statistic: {F:.2f} on {k} and {dfr} DF,  p-value: < 2.2e-16"]
    return "\n".join(L) + "\n"


def main():
    ensure_dirs()
    d = load_model_data()
    tr, te = d[d.Split == "Train"], d[d.Split == "Test"]
    Xtr, ytr, Xte, yte = tr[MODEL_STATS], tr[TARGET].values, te[MODEL_STATS], te[TARGET].values

    # 1. Hyperparameter search (5-fold CV on training set)
    grid = {"elasticnet__alpha": 10 ** np.arange(-4, 0.01, 0.25),
            "elasticnet__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0]}
    gs = GridSearchCV(en_pipeline(), grid, cv=KFold(5, shuffle=True, random_state=SEED),
                      scoring="neg_mean_squared_error").fit(Xtr, ytr)
    pd.DataFrame(gs.cv_results_)[["param_elasticnet__alpha", "param_elasticnet__l1_ratio",
                                  "mean_test_score", "std_test_score"]].to_csv(
        RESULTS / "elastic_net_tuning.csv", index=False)
    best = gs.best_params_
    print(f"Grid search best: alpha={best['elasticnet__alpha']:.5f}, l1_ratio={best['elasticnet__l1_ratio']}")
    print(f"Reported model uses: alpha={EN_ALPHA:.5f}, l1_ratio={EN_L1_RATIO}")

    # 2. Train/test evaluation
    model = en_pipeline().fit(Xtr, ytr)
    ptr, pte = model.predict(Xtr), model.predict(Xte)
    coefs = pd.DataFrame({"stat": MODEL_STATS, "coef": model[-1].coef_}).sort_values("coef", key=abs, ascending=False)
    coefs.to_csv(RESULTS / "elastic_net_coefs.csv", index=False)

    # 3. Repeated 5-fold CV (20 repeats) on the training set
    cv = cross_validate(en_pipeline(), Xtr, ytr, cv=RepeatedKFold(n_splits=5, n_repeats=20, random_state=SEED),
                        scoring=("r2", "neg_root_mean_squared_error"))

    # 4. Full-sample OLS (standardized)
    X, y = d[MODEL_STATS].values, d[TARGET].values
    b, se, se_hc3, e, dfr = ols(X, y)
    n, k = len(y), len(MODEL_STATS)
    yz = (y - y.mean()) / y.std(ddof=1)
    r2_full = 1 - (e @ e) / (yz @ yz)
    adj = 1 - (1 - r2_full) * (n - 1) / dfr
    F = (r2_full / k) / ((1 - r2_full) / dfr)
    t, t3 = b / se, b / se_hc3
    p, p3 = 2 * stats.t.sf(np.abs(t), dfr), 2 * stats.t.sf(np.abs(t3), dfr)
    names = ["(Intercept)"] + MODEL_STATS
    pd.DataFrame({"term": names, "beta": b, "se": se, "t": t, "p": p,
                  "se_hc3": se_hc3, "t_hc3": t3, "p_hc3": p3}).to_csv(RESULTS / "ols_coefficients.csv", index=False)
    (RESULTS / "ols_summary.txt").write_text(r_style_summary(names, b, se, t, p, e, dfr, r2_full, adj, F, k))

    # 5. Sensitivity: cross-validated R2 at other minutes cutoffs (repeated 5-fold, 20 repeats, full sample)
    ps = load_player_seasons()
    sens = []
    for cutoff in [250, 400, 500, 750, 1000, 1250]:
        s = add_impact_score(ps[ps.MP >= cutoff].dropna(subset=MODEL_STATS + ["OnCourt", "OnOff"]))
        sc = cross_validate(en_pipeline(), s[MODEL_STATS], s[TARGET], scoring="r2",
                            cv=RepeatedKFold(n_splits=5, n_repeats=20, random_state=SEED))["test_score"]
        sens.append({"min_mp": cutoff, "n": len(s), "cv_r2_mean": sc.mean(), "cv_r2_sd": sc.std()})
    pd.DataFrame(sens).to_csv(RESULTS / "cutoff_sensitivity.csv", index=False)

    metrics = {
        "n_train": len(tr), "n_test": len(te),
        "train_r2": r2(ytr, ptr), "test_r2": r2(yte, pte),
        "train_rmse": rmse(ytr, ptr), "test_rmse": rmse(yte, pte),
        "repeated_cv_r2_mean": float(cv["test_r2"].mean()), "repeated_cv_r2_sd": float(cv["test_r2"].std()),
        "repeated_cv_rmse_mean": float(-cv["test_neg_root_mean_squared_error"].mean()),
        "ols_r2": float(r2_full), "ols_adj_r2": float(adj), "ols_F": float(F), "ols_df": [k, int(dfr)],
        "ols_F_p": float(stats.f.sf(F, k, dfr)),
        "grid_best_alpha": float(best["elasticnet__alpha"]), "grid_best_l1_ratio": float(best["elasticnet__l1_ratio"]),
    }
    (RESULTS / "model_metrics.json").write_text(json.dumps(metrics, indent=2))
    for key, v in metrics.items():
        print(f"{key}: {v if isinstance(v, (int, list)) else round(v, 4)}")
    print(pd.DataFrame(sens).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
