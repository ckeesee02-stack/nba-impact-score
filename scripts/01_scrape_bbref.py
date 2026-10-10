"""Download Per 100 Possessions and Play-by-Play (+/-) league tables from Basketball-Reference.

Optional: the repo already ships data/player_seasons.csv. Run this only to rebuild from source.
Respects Sports Reference's rate limit (no more than 20 requests per minute).

Output: data/raw/per_poss_{year}.csv and data/raw/pbp_{year}.csv
"""
import re
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup

from common import RAW, SEASONS, ensure_dirs

BASE = "https://www.basketball-reference.com/leagues/NBA_{year}_{page}.html"
HEADERS = {"User-Agent": "Mozilla/5.0 (research script; contact via GitHub)"}
DELAY = 4.0  # seconds between requests

PER_POSS_MAP = {
    "name_display": "Player", "team_name_abbr": "Team", "mp": "MP",
    "fg_per_poss": "FG", "fga_per_poss": "FGA", "fg_pct": "FG%",
    "fg3_per_poss": "3P", "fg3a_per_poss": "3PA", "fg3_pct": "3P%",
    "fg2_per_poss": "2P", "fg2a_per_poss": "2PA", "fg2_pct": "2P%", "efg_pct": "eFG%",
    "ft_per_poss": "FT", "fta_per_poss": "FTA", "ft_pct": "FT%",
    "orb_per_poss": "ORB", "drb_per_poss": "DRB", "trb_per_poss": "TRB",
    "ast_per_poss": "AST", "stl_per_poss": "STL", "blk_per_poss": "BLK",
    "tov_per_poss": "TOV", "pf_per_poss": "PF", "pts_per_poss": "PTS",
}
PBP_MAP = {"name_display": "Player", "plus_minus_on": "OnCourt", "plus_minus_net": "OnOff"}


def fetch_table(year: int, page: str, table_id: str) -> pd.DataFrame:
    resp = requests.get(BASE.format(year=year, page=page), headers=HEADERS, timeout=30)
    resp.raise_for_status()
    html = re.sub(r"<!--|-->", "", resp.text)  # some tables live inside HTML comments
    table = BeautifulSoup(html, "html.parser").find("table", id=table_id)
    if table is None:
        raise ValueError(f"Table '{table_id}' not found for {year}")
    rows = []
    for tr in table.select("tbody tr"):
        if "thead" in (tr.get("class") or []):
            continue
        rec = {}
        for cell in tr.find_all(["th", "td"]):
            stat = cell.get("data-stat")
            if stat:
                rec[stat] = cell.get_text(strip=True)
                if stat == "name_display":
                    rec["PlayerID"] = cell.get("data-append-csv", "")
        if rec.get("name_display") and rec["name_display"] != "League Average":
            rows.append(rec)
    return pd.DataFrame(rows)


def main():
    ensure_dirs()
    for year in SEASONS:
        for page, table_id, cmap, name in [("per_poss", "per_poss", PER_POSS_MAP, "per_poss"),
                                           ("play-by-play", "pbp_stats", PBP_MAP, "pbp")]:
            out = RAW / f"{name}_{year}.csv"
            if out.exists():
                print(f"skip {out.name}")
                continue
            df = fetch_table(year, page, table_id)
            df = df[["PlayerID"] + [c for c in cmap if c in df.columns]].rename(columns=cmap)
            df.to_csv(out, index=False)
            print(f"saved {out.name} ({len(df)} rows)")
            time.sleep(DELAY)


if __name__ == "__main__":
    main()
