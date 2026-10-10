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
PANEL = "#242422"; EDGE = "#3a3a37"; TARGET = "#9085e9"; POS = "#3987e5"; NEG = "#d95926"

d = load_model_data()
y = d["ImpactScore"]
F = ['FG', '3PA', '3P%', '2P%', 'eFG%', 'FTA', 'FT%', 'ORB', 'DRB', 'AST', 'STL', 'BLK', 'TOV', 'PF']
r = pd.Series({f: d[f].corr(y) for f in F}).sort_values()

W, H = 16, 10.5
fig = plt.figure(figsize=(W, H), dpi=150, facecolor=BG)

def box(x, yy, w, h, fc=PANEL, ec=EDGE, rr=0.008):
    fig.patches.append(FancyBboxPatch((x, yy), w, h, boxstyle=f"round,pad=0,rounding_size={rr}",
                                      transform=fig.transFigure, facecolor=fc, edgecolor=ec, linewidth=1))

def style(ax):
    ax.set_facecolor(BG)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.tick_params(colors=T3, labelsize=10.5, length=0)

# ---- Header
fig.text(0.045, 0.945, "Under the Hood: The Data Behind Impact Score", color=T1, fontsize=32, fontweight="bold")
fig.text(0.045, 0.908, "Exploratory look at the dataset behind The Science of Positive Contribution  ·  NBA 2016-17 to 2025-26",
         color=T2, fontsize=14)

# ---- KPI row
kpis = [("3,532", "player-seasons"), ("888", "unique players"), ("10", "seasons, ~350 each"),
        ("500+", "minutes to qualify"), ("14", "stats in the model"), ("r = .60", "OnCourt vs. On-Off")]
kw, kx, ky, kh = 0.143, 0.045, 0.800, 0.080
for i, (v, l) in enumerate(kpis):
    x = kx + i * (kw + 0.0084)
    box(x, ky, kw, kh)
    fig.text(x + 0.012, ky + 0.040, v, color=TARGET if i == 5 else T1, fontsize=22, fontweight="bold")
    fig.text(x + 0.012, ky + 0.016, l, color=T2, fontsize=11.5)

# ---- Panel 1: Impact Score distribution
fig.text(0.045, 0.745, "1. Impact Score is close to a bell curve", color=T1, fontsize=15, fontweight="bold")
fig.text(0.045, 0.722, "Most players sit near average; only 14 are 3+ SD from it", color=T2, fontsize=11.5)
a1 = fig.add_axes([0.055, 0.36, 0.27, 0.34]); style(a1)
bins = np.linspace(-3.6, 3.6, 37)
a1.hist(y, bins=bins, color=TARGET, edgecolor=BG, linewidth=1.5)
xs = np.linspace(-3.6, 3.6, 200)
a1.plot(xs, len(y) * (bins[1] - bins[0]) * np.exp(-xs**2 / 2) / np.sqrt(2 * np.pi), color=T2, lw=1.5, ls="--")
a1.yaxis.grid(True, color=GRID); a1.set_axisbelow(True)
a1.set_xlabel("Impact Score (standard deviations)", color=T3, fontsize=11)
a1.set_xlim(-3.8, 3.8)
a1.annotate("Gobert '21\n+3.4", xy=(3.44, 3), xytext=(2.55, 150), color=T1, fontsize=10.5, ha="center",
            arrowprops=dict(arrowstyle="-", color=T3, lw=1))
a1.annotate("B. Williams '22\n-3.4", xy=(-3.43, 3), xytext=(-2.5, 150), color=T1, fontsize=10.5, ha="center",
            arrowprops=dict(arrowstyle="-", color=T3, lw=1))
a1.text(1.25, a1.get_ylim()[1] * 0.93, "dashed = perfect\nnormal curve", color=T3, fontsize=9.5)

# ---- Panel 2: OnCourt vs On-Off
fig.text(0.375, 0.745, "2. The two inputs agree, but not perfectly", color=T1, fontsize=15, fontweight="bold")
fig.text(0.375, 0.722, "r = .60: related, but each adds its own information", color=T2, fontsize=11.5)
a2 = fig.add_axes([0.385, 0.36, 0.25, 0.34]); style(a2)
a2.scatter(d["OnCourt"], d["OnOff"], s=7, color=POS, alpha=0.35, linewidths=0)
m, b = np.polyfit(d["OnCourt"], d["OnOff"], 1)
xx = np.array([-22, 21]); a2.plot(xx, m * xx + b, color=T1, lw=2)
a2.axhline(0, color=GRID, lw=1); a2.axvline(0, color=GRID, lw=1)
a2.set_xlim(-23, 23); a2.set_ylim(-23, 23)
a2.set_xlabel("OnCourt Net Rating (per 100)", color=T3, fontsize=11)
a2.set_ylabel("On-Off Net Rating (per 100)", color=T3, fontsize=11)
for nm, lab, dx, dy in [("Nikola Jokić", "Jokić '23", -15, -0.5), ("Chris Paul", "CP3 '17", -14, -4.5)]:
    row = d[(d.Player == nm)].sort_values("ImpactScore").iloc[-1]
    a2.annotate(lab, xy=(row.OnCourt, row["OnOff"]), xytext=(row.OnCourt + dx, row["OnOff"] + dy),
                color=T1, fontsize=10, arrowprops=dict(arrowstyle="-", color=T3, lw=1))

# ---- Panel 3: correlations with target
fig.text(0.685, 0.745, "3. One-on-one link with Impact Score", color=T1, fontsize=15, fontweight="bold")
fig.text(0.685, 0.722, "Raw correlation, before controlling for other stats", color=T2, fontsize=11.5)
a3 = fig.add_axes([0.725, 0.36, 0.23, 0.34]); style(a3)
cols = [POS if v > 0 else NEG for v in r.values]
a3.barh(range(len(r)), r.values, color=cols, height=0.7)
a3.set_yticks(range(len(r))); a3.set_yticklabels(r.index, color=T1, fontsize=11)
for i, v in enumerate(r.values):
    a3.text(v + (0.008 if v >= 0 else -0.008), i, f"{v:+.2f}".replace("+0", "+").replace("-0", "-"),
            va="center", ha="left" if v >= 0 else "right", color=T2, fontsize=9.5)
a3.axvline(0, color=T3, lw=1); a3.set_xlim(-0.16, 0.38)
a3.xaxis.grid(True, color=GRID); a3.set_axisbelow(True)
a3.set_xlabel("Correlation (r)", color=T3, fontsize=11)
i_tov = list(r.index).index("TOV")
a3.text(0.10, i_tov - 0.05, "TOV* ~0 alone", color=NEG, fontsize=9.5, va="center")

# ---- Bottom boxes
bx_y, bx_h = 0.085, 0.195
box(0.045, bx_y, 0.44, bx_h)
fig.text(0.06, bx_y + bx_h - 0.035, "For everyone", color=T1, fontsize=15, fontweight="bold")
gen = ["Shooting efficiency has the strongest one-on-one link to impact.",
       "Turnovers look neutral alone, because high-usage stars commit more.",
       "Players who play more minutes tend to score higher (r = .31).",
       "Extreme seasons are rare: about 1 in 250 is 3+ SD from average."]
for i, t in enumerate(gen):
    fig.text(0.06, bx_y + bx_h - 0.068 - i * 0.031, "•  " + t, color=T2, fontsize=12)

box(0.51, bx_y, 0.445, bx_h)
fig.text(0.525, bx_y + bx_h - 0.035, "For the technical crowd", color=T1, fontsize=15, fontweight="bold")
tech = [("Target:", " skew -0.03, kurtosis 0.13; inputs both |skew| < 0.1"),
        ("Features:", " BLK (1.9), ORB (1.5), FTA (1.4) right-skewed; 3P%, FT% left"),
        ("Missing:", " 94 rows dropped (0 3PA, so 3P% undefined)"),
        ("Collinearity:", " 8 stats dropped at VIF > 10; all kept VIF < 4"),
        ("Caveat:", " rows not independent (888 players across seasons)")]
for i, (k, t) in enumerate(tech):
    yy = bx_y + bx_h - 0.068 - i * 0.026
    tx = fig.text(0.525, yy, k, color=TARGET, fontsize=11.5, fontweight="bold"); fig.canvas.draw()
    x1 = tx.get_window_extent().transformed(fig.transFigure.inverted()).x1
    fig.text(x1, yy, t, color=T2, fontsize=11.5)

# ---- Footer
fig.text(0.045, 0.035, "Impact Score = average of z-scored OnCourt and On-Off Net Rating per 100 possessions. "
         "*TOV turns negative once usage is held constant (see model). Data: Basketball-Reference.",
         color=T3, fontsize=10.5)
fig.text(0.955, 0.032, "@HoopKeesee", color=T1, fontsize=16, fontweight="bold", ha="right")

out = FIGURES / "impact_score_eda.png"
fig.savefig(out, facecolor=BG, dpi=150)
print(out)
