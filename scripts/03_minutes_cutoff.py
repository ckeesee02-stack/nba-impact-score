"""Minutes-cutoff analysis: year-to-year reliability of Impact Score at each minimum-minutes threshold.

For each cutoff, Impact Score is recomputed within that sample and correlated across consecutive
seasons for the same player (both seasons must clear the cutoff). 95% CI via Fisher z.

Output: results/minutes_cutoff.csv
"""
import numpy as np
import pandas as pd

from common import RESULTS, add_impact_score, ensure_dirs, load_player_seasons


def main():
    ensure_dirs()
    ps = load_player_seasons().dropna(subset=["OnCourt", "OnOff"])
    ps["yr"] = ps["Season"].str[:4].astype(int)
    rows = []
    for cutoff in range(100, 1601, 50):
        d = add_impact_score(ps[ps.MP >= cutoff])
        a = d[["PlayerID", "yr", "ImpactScore"]]
        b = a.copy()
        b["yr"] -= 1
        pairs = a.merge(b, on=["PlayerID", "yr"], suffixes=("_t", "_t1"))
        r = pairs["ImpactScore_t"].corr(pairs["ImpactScore_t1"])
        se = 1 / np.sqrt(len(pairs) - 3)
        rows.append({"min_mp": cutoff, "player_seasons": len(d), "pairs": len(pairs), "r": r,
                     "ci_low": np.tanh(np.arctanh(r) - 1.96 * se),
                     "ci_high": np.tanh(np.arctanh(r) + 1.96 * se)})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "minutes_cutoff.csv", index=False)
    print(out[out.min_mp.isin([100, 400, 500, 1000, 1500])].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
