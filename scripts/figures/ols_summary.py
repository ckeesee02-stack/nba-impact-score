"""Render results/ols_summary.txt as an R-console style image."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import FIGURES, RESULTS, ensure_dirs  # noqa: E402

ensure_dirs()
txt = (RESULTS / "ols_summary.txt").read_text().rstrip("\n")
lines = txt.split("\n")
w_in, h_in = max(map(len, lines)) * 0.118 + 0.6, len(lines) * 0.255 + 0.6
fig = plt.figure(figsize=(w_in, h_in), dpi=200, facecolor="#ececec")
fig.text(0.3 / w_in, 1 - 0.3 / h_in, txt, family="DejaVu Sans Mono", fontsize=13,
         va="top", ha="left", color="#1a1a1a", linespacing=1.45)
out = FIGURES / "ols_summary.png"
fig.savefig(out, facecolor="#ececec")
print(out)
