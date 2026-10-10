import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch
import pandas as pd, numpy as np, glob

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import FIGURES, RESULTS, ensure_dirs, load_model_data  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import set_font  # noqa: E402

set_font()
ensure_dirs()
BG = "#1a1a19"; T1 = "#ffffff"; T2 = "#c3c2b7"; T3 = "#8a8983"; GRID = "#2e2e2c"
PANEL = "#242422"; EDGE = "#3a3a37"; TARGET = "#9085e9"; LINE = "#3987e5"; BAR = "#4a4a46"

R = pd.read_csv(RESULTS / "minutes_cutoff.csv").rename(columns={"min_mp": "T", "player_seasons": "n", "ci_low": "lo", "ci_high": "hi"})
S = pd.read_csv(RESULTS / "cutoff_sensitivity.csv")
r500 = R[R["T"] == 500].iloc[0]

W, H = 16, 10
fig = plt.figure(figsize=(W, H), dpi=150, facecolor=BG)
def box(x, y, w, h):
    fig.patches.append(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.008",
                                      transform=fig.transFigure, facecolor=PANEL, edgecolor=EDGE, linewidth=1))
def style(ax):
    ax.set_facecolor(BG)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.tick_params(colors=T3, labelsize=11, length=0)

fig.text(0.045, 0.94, "Why 500 Minutes?", color=T1, fontsize=34, fontweight="bold")
fig.text(0.045, 0.902, "Choosing the minimum minutes for the Impact Score model  ·  NBA 2016-17 to 2025-26",
         color=T2, fontsize=14.5)

# Top: reliability
fig.text(0.045, 0.835, "How repeatable is Impact Score from one season to the next?", color=T1, fontsize=16, fontweight="bold")
fig.text(0.045, 0.808, "Year-to-year correlation of the same player's Impact Score, by minimum minutes (shaded = 95% CI)",
         color=T2, fontsize=12)
ax = fig.add_axes([0.075, 0.455, 0.56, 0.33]); style(ax)
ax.axvspan(100, 400, color="#232330", zorder=0); ax.axvspan(400, 1000, color="#1f2a24", zorder=0)
ax.axvspan(1250, 1600, color="#2a2420", zorder=0)
ax.fill_between(R["T"], R.lo, R.hi, color=LINE, alpha=0.18, linewidth=0)
ax.plot(R["T"], R.r, color=LINE, lw=2.5)
ax.axvline(500, color=TARGET, lw=2, ls="--")
ax.scatter([500], [r500.r], s=90, color=TARGET, zorder=5, edgecolor=BG, linewidth=2)
ax.text(520, r500.r - 0.03, f"500 MP: r = {r500.r:.2f}", color=TARGET, fontsize=12.5, fontweight="bold")
ax.set_xlim(75, 1625); ax.set_ylim(0.28, 0.52)
ax.set_xticks(range(100, 1601, 100)); ax.set_xlabel("Minimum minutes played in the season", color=T3, fontsize=11.5)
ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
ax.set_ylabel("Year-to-year r", color=T3, fontsize=11.5)
for x0, x1, lab in [(100, 400, "Steep gains\n+.06"), (400, 1000, "Plateau\n+.02 over 600 more minutes"),
                    (1250, 1600, "Rises again, but only\nby dropping bench players")]:
    ax.text((x0 + x1) / 2, 0.495, lab, color=T2, fontsize=11, ha="center", va="top", linespacing=1.3)

# Bottom: sample size
fig.text(0.045, 0.352, "What each cutoff costs in sample size", color=T1, fontsize=16, fontweight="bold")
ax2 = fig.add_axes([0.075, 0.12, 0.56, 0.21]); style(ax2)
ax2.bar(R["T"], R.n, width=38, color=[TARGET if t == 500 else BAR for t in R["T"]])
ax2.set_xlim(75, 1625); ax2.set_ylim(0, 5200)
ax2.set_xticks(range(100, 1601, 100))
ax2.yaxis.grid(True, color=GRID); ax2.set_axisbelow(True)
ax2.set_yticks([0, 1000, 2000, 3000, 4000, 5000]); ax2.set_yticklabels(["0", "1k", "2k", "3k", "4k", "5k"])
ax2.set_ylabel("Player-seasons", color=T3, fontsize=11.5)
ax2.set_xlabel("Minimum minutes played in the season", color=T3, fontsize=11.5)
ax2.text(500, r500.n + 150, f"{int(r500.n):,}", color=TARGET, fontsize=12, fontweight="bold", ha="center")
n1000 = int(R[R["T"] == 1000].n.iloc[0]); n1500 = int(R[R["T"] == 1500].n.iloc[0])
ax2.text(1000, n1000 + 150, f"{n1000:,}", color=T2, fontsize=11, ha="center")
ax2.text(1500, n1500 + 150, f"{n1500:,}", color=T2, fontsize=11, ha="center")

# Right: explanation
bx, by, bw, bh = 0.68, 0.12, 0.275, 0.665
box(bx, by, bw, bh)
fig.text(bx + 0.018, by + bh - 0.045, "The tradeoff", color=T1, fontsize=17, fontweight="bold")
blocks = [
    ("Too low (under 400)", ["Small-minute samples are noisier.", "Impact Score carries over much", "less reliably year to year."]),
    ("500 minutes", ["Reliability has leveled off", f"(r = {r500.r:.2f}) while keeping 77% of", "all player-seasons with 100+ MP."]),
    ("Higher cutoffs", ["Doubling to 1,000 adds only +.02", "and cuts the sample by 26%.", "Past 1,250, gains come from keeping", "only starters, which biases the model."]),
]
y = by + bh - 0.095
for title, lines in blocks:
    fig.text(bx + 0.018, y, title, color=TARGET if title == "500 minutes" else T1, fontsize=13.5, fontweight="bold")
    y -= 0.033
    for l in lines:
        fig.text(bx + 0.018, y, l, color=T2, fontsize=12.2); y -= 0.028
    y -= 0.022
fig.text(bx + 0.018, by + 0.035, f"Model fit was also stable across cutoffs\n(cross-validated R² {S.cv_r2_mean.min():.2f} to {S.cv_r2_mean.max():.2f}).", color=T3, fontsize=11.5,
         linespacing=1.4)

fig.text(0.045, 0.035, "Year-to-year r: correlation between a player's Impact Score in consecutive seasons, both seasons above the cutoff. "
         "CI via Fisher z. Data: Basketball-Reference.", color=T3, fontsize=10.5)
fig.text(0.955, 0.032, "@HoopKeesee", color=T1, fontsize=16, fontweight="bold", ha="right")
out = FIGURES / "minutes_cutoff.png"
fig.savefig(out, facecolor=BG, dpi=150); print(out)
