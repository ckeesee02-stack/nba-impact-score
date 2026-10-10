import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch
import glob
import json
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import FIGURES, RESULTS, ensure_dirs, load_model_data  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import set_font  # noqa: E402

set_font()
ensure_dirs()

BG = "#1a1a19"; T1 = "#ffffff"; T2 = "#c3c2b7"; T3 = "#8a8983"; GRID = "#2e2e2c"; EMPTY = "#2f2f2d"
PANEL = "#242422"; EDGE = "#3a3a37"; TARGET = "#9085e9"
C = {"Shooting efficiency": "#3987e5", "Playmaking": "#d95926",
     "Defense": "#199e70", "Volume & rebounding": "#c98500"}

GROUP = {"eFG%": "Shooting efficiency", "2P%": "Shooting efficiency", "FT%": "Shooting efficiency",
         "3P%": "Shooting efficiency", "AST": "Playmaking", "TOV": "Playmaking", "STL": "Defense",
         "BLK": "Defense", "PF": "Defense", "FTA": "Volume & rebounding", "FG": "Volume & rebounding",
         "DRB": "Volume & rebounding", "ORB": "Volume & rebounding", "3PA": "Volume & rebounding"}
rw = pd.read_csv(RESULTS / "relative_weights.csv")
data = [(s, round(v, 1), GROUP[s]) for s, v in zip(rw.stat, rw.pct_of_r2)]
R2 = 100 * json.loads((RESULTS / "model_metrics.json").read_text())["ols_r2"]

W, H = 16, 10.5
fig = plt.figure(figsize=(W, H), dpi=150, facecolor=BG)

def box(x, y, w, h, fc=PANEL, ec=EDGE, lw=1, r=0.008):
    fig.patches.append(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                      transform=fig.transFigure, facecolor=fc, edgecolor=ec, linewidth=lw))

# ---- Header
fig.text(0.045, 0.940, "The Science of Positive Contribution", color=T1, fontsize=36, fontweight="bold")
fig.text(0.045, 0.902, "Which box-score stats explain a player's Impact Score?  ·  NBA 2016-17 to 2025-26  ·  3,532 player-seasons (500+ minutes)",
         color=T2, fontsize=14.5)

# ---- Target band: OnCourt + On-Off = Impact Score
by, bh = 0.792, 0.085
box(0.045, by, 0.91, bh)
fig.text(0.062, by + bh / 2, "THE TARGET", color=T3, fontsize=11.5, fontweight="bold", va="center")
# pill 1
box(0.150, by + 0.014, 0.235, bh - 0.028, fc="#2c2c2a", ec="#4a4a46")
fig.text(0.162, by + 0.052, "OnCourt Net Rating", color=T1, fontsize=14.5, fontweight="bold")
fig.text(0.162, by + 0.026, "Team +/- per 100 poss with him playing", color=T2, fontsize=11.5)
fig.text(0.398, by + bh / 2, "+", color=T2, fontsize=26, fontweight="bold", va="center", ha="center")
# pill 2
box(0.412, by + 0.014, 0.255, bh - 0.028, fc="#2c2c2a", ec="#4a4a46")
fig.text(0.424, by + 0.052, "On-Off Net Rating", color=T1, fontsize=14.5, fontweight="bold")
fig.text(0.424, by + 0.026, "How much better the team is with him on vs. off", color=T2, fontsize=11.5)
fig.text(0.681, by + bh / 2, "=", color=T2, fontsize=26, fontweight="bold", va="center", ha="center")
# target pill
box(0.697, by + 0.012, 0.243, bh - 0.024, fc=TARGET, ec=TARGET, r=0.01)
fig.text(0.818, by + 0.050, "IMPACT SCORE", color="#16133a", fontsize=19, fontweight="bold", ha="center")
fig.text(0.818, by + 0.024, "both standardized, then averaged", color="#16133a", fontsize=11.5, ha="center")

# ---- Left: relative weights bars
fig.text(0.045, 0.735, "What the box score explains, split by stat", color=T1, fontsize=17, fontweight="bold")
lx = 0.045
for g, col in C.items():
    box(lx, 0.700, 0.010, 0.015, fc=col, ec=col, r=0.002)
    fig.text(lx + 0.015, 0.7015, g, color=T2, fontsize=12)
    lx += 0.015 + len(g) * 0.0069 + 0.02

ax = fig.add_axes([0.075, 0.16, 0.47, 0.52], facecolor=BG)
ys = list(range(len(data)))[::-1]
for y, (s, v, g) in zip(ys, data):
    ax.barh(y, v, height=0.72, color=C[g], linewidth=0)
    ax.text(v + 0.35, y, f"{v:.1f}%", va="center", ha="left", color=T1 if v >= 7 else T2,
            fontsize=13, fontweight="semibold" if v >= 7 else "normal")
ax.set_yticks(ys); ax.set_yticklabels([d[0] + ("*" if d[0] == "TOV" else "") for d in data], color=T1, fontsize=14)
ax.tick_params(axis="y", length=0, pad=10); ax.tick_params(axis="x", colors=T3, labelsize=11, length=0)
ax.set_xlim(0, 25); ax.set_xticks([0, 5, 10, 15, 20, 25]); ax.set_xticklabels([f"{t}%" for t in [0, 5, 10, 15, 20, 25]])
ax.xaxis.grid(True, color=GRID, linewidth=1); ax.set_axisbelow(True)
for sp in ax.spines.values(): sp.set_visible(False)
ax.set_ylim(-0.7, len(data) - 0.3)
ax.set_xlabel("Relative weight: share of the explained 26% (bars add to 100%)", color=T3, fontsize=11.5, labelpad=8)

# ---- Right: R² waffle
_x=0.63
for _t,_c in [("The box score explains ",T1),("26%",TARGET),(" of Impact Score",T1)]:
    _tx=fig.text(_x,0.735,_t,color=_c,fontsize=17,fontweight="bold"); fig.canvas.draw()
    _x=_tx.get_window_extent().transformed(fig.transFigure.inverted()).x1
fig.text(0.63, 0.704, "R² = 0.26   ·   each square = 1% of the differences between players", color=T2, fontsize=12)

wf = fig.add_axes([0.63, 0.36, 0.30, 0.33], facecolor=BG)
cells = []
for g in C:
    cells += [C[g]] * round(sum(v for _, v, gg in data if gg == g) / 100 * R2)
cells += [EMPTY] * (100 - len(cells))
for i, col in enumerate(cells):
    r, c = divmod(i, 10)
    wf.add_patch(FancyBboxPatch((c + 0.08, 9 - r + 0.08), 0.84, 0.84,
                                boxstyle="round,pad=0,rounding_size=0.12", color=col, linewidth=0))
wf.set_xlim(0, 10); wf.set_ylim(0, 10); wf.set_aspect("equal"); wf.axis("off")
wx = wf.get_position().x0 + (wf.get_position().width - wf.get_position().height * H / W) / 2
fig.text(0.63, 0.335, "Gray = lineups, defense the box score can't see, and noise.", color=T2, fontsize=12)

# ---- How to read
box(0.63, 0.112, 0.31, 0.203)
fig.text(0.645, 0.283, "How to read this", color=T1, fontsize=15, fontweight="bold")
lines = [("R² = 0.26:", " box-score stats explain about a quarter"),
         ("", "of why some players' Impact Scores are higher."),
         ("Relative weights:", " split that quarter across the stats."),
         ("", "eFG% alone owns 21.9% of it, the biggest slice."),
         ("Takeaway:", " efficient shooting, steals, and assists")]
yy = 0.252
for b, t in lines:
    if b:
        tx = fig.text(0.645, yy, b, color=TARGET, fontsize=12.5, fontweight="bold")
        fig.canvas.draw()
        bb = tx.get_window_extent().transformed(fig.transFigure.inverted())
        fig.text(bb.x1, yy, t, color=T2, fontsize=12.5)
    else:
        fig.text(0.645, yy, t, color=T2, fontsize=12.5)
    yy -= 0.0245
fig.text(0.645, yy, "drive positive contribution most.", color=T2, fontsize=12.5)

# ---- Footer
fig.text(0.045, 0.084, "*TOV has a negative relationship: with other stats held equal, more turnovers means a lower Impact Score.", color=T2, fontsize=11.5)
fig.text(0.045, 0.030,
         "Method: Per 100 Possessions stats (z-scored, overlapping stats removed) predicting Impact Score with an elastic net regression; "
         "cross-validated R² 0.25, test R² 0.26.\nRelative weights (Johnson). Data: Basketball-Reference.",
         color=T3, fontsize=10.5, linespacing=1.5)
fig.text(0.955, 0.036, "@HoopKeesee", color=T1, fontsize=16, fontweight="bold", ha="right")

out = FIGURES / "positive_contribution.png"
fig.savefig(out, facecolor=BG, dpi=150)
print(out)
