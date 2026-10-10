"""Collinearity control: iteratively drop the highest-VIF stat until every VIF <= threshold.

Computed on the training set only. Thresholds of 10 and 5 produce identical results.

Output: results/vif_removed.csv, results/vif_final.csv
"""
import pandas as pd

from common import ALL_STATS, MODEL_STATS, RESULTS, VIF_THRESHOLD, ensure_dirs, load_model_data, vif


def prune(train: pd.DataFrame, threshold: float):
    feats, removed = ALL_STATS.copy(), []
    while True:
        v = vif(train[feats].values)
        i = v.argmax()
        if v[i] <= threshold:
            return feats, removed, dict(zip(feats, v))
        removed.append({"order": len(removed) + 1, "stat": feats[i], "vif_when_removed": v[i]})
        feats.pop(i)


def main():
    ensure_dirs()
    train = load_model_data().query("Split == 'Train'")
    kept, removed, final = prune(train, VIF_THRESHOLD)
    kept5, _, _ = prune(train, 5)

    pd.DataFrame(removed).to_csv(RESULTS / "vif_removed.csv", index=False)
    pd.DataFrame({"stat": list(final), "vif": list(final.values())}).to_csv(RESULTS / "vif_final.csv", index=False)

    print(pd.DataFrame(removed).round(1).to_string(index=False))
    print(f"Kept {len(kept)} stats; max VIF {max(final.values()):.2f}")
    print(f"Threshold 5 gives the same set: {set(kept5) == set(kept)}")
    assert set(kept) == set(MODEL_STATS), "Kept stats differ from MODEL_STATS in common.py"


if __name__ == "__main__":
    main()
