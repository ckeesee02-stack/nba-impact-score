"""Build the modeling dataset.

1. If data/raw/ has scraped files, rebuild data/player_seasons.csv from them
   (one row per player-season; traded players use the season-total row, which is listed first).
   Otherwise use the shipped data/player_seasons.csv.
2. Keep player-seasons with 500+ minutes and no missing values (drops players with 0 3PA, since 3P% is undefined).
3. Compute Impact Score = z( mean( z(OnCourt), z(OnOff) ) ).
4. Assign an 80/20 train/test split if one is not already stored.

Output: data/model_data.csv
"""
import pandas as pd
from sklearn.model_selection import train_test_split

from common import (ALL_STATS, DATA, MIN_MP, RAW, SEASONS, SEED, TEST_SIZE,
                    add_impact_score, load_player_seasons)


def rebuild_from_raw() -> pd.DataFrame:
    frames = []
    for year in SEASONS:
        pp = pd.read_csv(RAW / f"per_poss_{year}.csv").drop_duplicates("PlayerID", keep="first")
        pbp = pd.read_csv(RAW / f"pbp_{year}.csv").drop_duplicates("PlayerID", keep="first")
        m = pp.merge(pbp[["PlayerID", "OnCourt", "OnOff"]], on="PlayerID", how="inner")
        m.insert(0, "Season", f"{year - 1}-{str(year)[2:]}")
        frames.append(m)
    df = pd.concat(frames, ignore_index=True)
    num = ["MP"] + ALL_STATS + ["OnCourt", "OnOff"]
    df[num] = df[num].apply(pd.to_numeric, errors="coerce")
    df = df[df.MP >= 100].copy()
    df["Split"] = pd.NA
    cols = ["Season", "Player", "PlayerID", "Team", "MP"] + ALL_STATS + ["OnCourt", "OnOff", "Split"]
    return df[cols]


def main():
    have_raw = all((RAW / f"{p}_{y}.csv").exists() for y in SEASONS for p in ("per_poss", "pbp"))
    if have_raw:
        ps = rebuild_from_raw()
        ps.to_csv(DATA / "player_seasons.csv", index=False)
        print(f"Rebuilt player_seasons.csv from raw files ({len(ps)} rows)")
    else:
        ps = load_player_seasons()
        print(f"Using shipped player_seasons.csv ({len(ps)} rows)")

    qualified = ps[ps.MP >= MIN_MP]
    d = qualified.dropna(subset=ALL_STATS + ["OnCourt", "OnOff"]).copy()
    print(f"{len(qualified)} player-seasons with {MIN_MP}+ MP; "
          f"{len(qualified) - len(d)} dropped for missing values; {len(d)} remain")

    d = add_impact_score(d)

    if d["Split"].isna().any():
        tr, _ = train_test_split(d.index, test_size=TEST_SIZE, random_state=SEED)
        d["Split"] = "Test"
        d.loc[tr, "Split"] = "Train"
        print("Assigned new 80/20 split (seed 42)")
    print(d["Split"].value_counts().to_dict())

    d.to_csv(DATA / "model_data.csv", index=False)


if __name__ == "__main__":
    main()
