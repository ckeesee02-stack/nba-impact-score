"""Exploratory data analysis on the modeling sample.

Outputs: results/eda_summary.json, results/feature_summary.csv
"""
import json

import numpy as np
import pandas as pd

from common import MIN_MP, MODEL_STATS, RESULTS, TARGET, ensure_dirs, load_model_data, load_player_seasons


def main():
    ensure_dirs()
    d = load_model_data()
    ps = load_player_seasons()
    y = d[TARGET]

    feats = pd.DataFrame({
        "mean": d[MODEL_STATS].mean(), "sd": d[MODEL_STATS].std(), "skew": d[MODEL_STATS].skew(),
        "r_with_target": [d[f].corr(y) for f in MODEL_STATS],
        "n_beyond_3sd": [int(((d[f] - d[f].mean()).abs() > 3 * d[f].std()).sum()) for f in MODEL_STATS],
    }).rename_axis("stat").sort_values("r_with_target", ascending=False)
    feats.to_csv(RESULTS / "feature_summary.csv")

    top = d.sort_values(TARGET)
    summary = {
        "qualified_player_seasons": int((ps.MP >= MIN_MP).sum()),
        "model_player_seasons": len(d),
        "unique_players": int(d.PlayerID.nunique()),
        "per_season": d.Season.value_counts().sort_index().to_dict(),
        "minutes": d.MP.describe().round(0).to_dict(),
        "oncourt": {"mean": d.OnCourt.mean(), "sd": d.OnCourt.std(), "skew": d.OnCourt.skew()},
        "onoff": {"mean": d.OnOff.mean(), "sd": d.OnOff.std(), "skew": d.OnOff.skew()},
        "r_oncourt_onoff": d.OnCourt.corr(d.OnOff),
        "target_skew": y.skew(), "target_kurtosis": y.kurt(),
        "target_beyond_3sd": int((y.abs() > 3).sum()),
        "r_target_minutes": y.corr(d.MP),
        "top3": top.tail(3)[["Season", "Player", TARGET]].round(2).to_dict("records")[::-1],
        "bottom": top.head(1)[["Season", "Player", TARGET]].round(2).to_dict("records"),
    }
    (RESULTS / "eda_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=lambda v: round(float(v), 3)))
    print(feats.round(3).to_string())


if __name__ == "__main__":
    main()
