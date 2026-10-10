"""Shared constants, paths, and helpers for the Impact Score project."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

SEASONS = list(range(2017, 2027))  # Basketball-Reference season end years: 2016-17 ... 2025-26
MIN_MP = 500
SEED = 42
TEST_SIZE = 0.20
VIF_THRESHOLD = 10  # a threshold of 5 removes the same stats

# All Per 100 Possessions stats (excluding G, GS, MP, ORtg, DRtg)
ALL_STATS = ["FG", "FGA", "FG%", "3P", "3PA", "3P%", "2P", "2PA", "2P%", "eFG%",
             "FT", "FTA", "FT%", "ORB", "DRB", "TRB", "AST", "STL", "BLK", "TOV", "PF", "PTS"]

# Stats retained after VIF pruning (see 04_collinearity.py)
MODEL_STATS = ["FG", "3PA", "3P%", "2P%", "eFG%", "FTA", "FT%",
               "ORB", "DRB", "AST", "STL", "BLK", "TOV", "PF"]

# Elastic net hyperparameters selected by 5-fold CV on the training set
EN_ALPHA = 0.0017782794100389228   # 10 ** -2.75
EN_L1_RATIO = 0.95

TARGET = "ImpactScore"


def zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std(ddof=1)


def add_impact_score(df: pd.DataFrame) -> pd.DataFrame:
    """Impact Score = z( mean( z(OnCourt), z(OnOff) ) ), computed within the given sample."""
    df = df.copy()
    df["zOnCourt"] = zscore(df["OnCourt"])
    df["zOnOff"] = zscore(df["OnOff"])
    df[TARGET] = zscore((df["zOnCourt"] + df["zOnOff"]) / 2)
    return df


def load_player_seasons() -> pd.DataFrame:
    return pd.read_csv(DATA / "player_seasons.csv")


def load_model_data() -> pd.DataFrame:
    path = DATA / "model_data.csv"
    if not path.exists():
        raise FileNotFoundError("Run scripts/02_build_dataset.py first.")
    return pd.read_csv(path)


def vif(X: np.ndarray) -> np.ndarray:
    """Variance inflation factors = diagonal of the inverse correlation matrix."""
    return np.diag(np.linalg.inv(np.corrcoef(X, rowvar=False)))


def relative_weights(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Johnson (2000) relative weights. Returns raw weights that sum to the OLS R-squared."""
    Z = (X - X.mean(0)) / X.std(0, ddof=1)
    yz = (y - y.mean()) / y.std(ddof=1)
    n = len(y)
    R = np.corrcoef(Z, rowvar=False)
    rxy = Z.T @ yz / (n - 1)
    evals, evecs = np.linalg.eigh(R)
    Lam = evecs @ np.diag(np.sqrt(evals)) @ evecs.T
    beta = np.linalg.solve(Lam, rxy)
    return (Lam ** 2) @ (beta ** 2)


def ensure_dirs():
    for p in (RAW, RESULTS, FIGURES):
        p.mkdir(parents=True, exist_ok=True)
