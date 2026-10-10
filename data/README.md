# Data

## Source

All data comes from [Basketball-Reference](https://www.basketball-reference.com) (Sports Reference LLC):

- `https://www.basketball-reference.com/leagues/NBA_{year}_per_poss.html` (Per 100 Possessions)
- `https://www.basketball-reference.com/leagues/NBA_{year}_play-by-play.html` (Play-by-Play, +/- per 100 possessions)

for season end years 2017 through 2026 (2016-17 to 2025-26 regular seasons). Use of this data is subject to the [Sports Reference terms of use](https://www.sports-reference.com/termsofuse.html). Please credit Basketball-Reference if you reuse it.

The CSV files described below are **not included** in this repository, because the Sports Reference terms of use restrict redistributing their data. Run `python scripts/run_all.py --scrape` to download the data and build them locally.

## `player_seasons.csv`

One row per player-season with at least 100 minutes played (4,712 rows in the original download). Traded players appear once, using their season-total row.

| Column | Description |
|---|---|
| Season | Season label, e.g. `2016-17` |
| Player | Player name |
| PlayerID | Basketball-Reference player ID (e.g. `jokicni01`) |
| Team | Team abbreviation (`2TM` / `3TM` / `TOT` for traded players) |
| MP | Minutes played |
| FG, FGA, 3P, 3PA, 2P, 2PA, FT, FTA | Makes and attempts per 100 team possessions |
| FG%, 3P%, 2P%, eFG%, FT% | Shooting percentages (blank when the player had no attempts) |
| ORB, DRB, TRB, AST, STL, BLK, TOV, PF, PTS | Per 100 team possessions |
| OnCourt | Team net rating per 100 possessions with the player on the floor |
| OnOff | OnCourt minus team net rating with the player off the floor |
| Split | `Train` / `Test` assignment used in the reported model (blank for rows outside the modeling sample). A fresh download draws a new split (seed 42). |

## `model_data.csv` (generated)

Created by `scripts/02_build_dataset.py`: rows with 500+ minutes and no missing values (3,532), plus:

| Column | Description |
|---|---|
| zOnCourt, zOnOff | Standardized OnCourt and OnOff within the modeling sample |
| ImpactScore | Standardized average of zOnCourt and zOnOff (the model target) |
