"""Shared font setup: uses Inter if installed (https://rsms.me/inter/), otherwise the default sans-serif."""
import glob

import matplotlib.pyplot as plt
from matplotlib import font_manager as fm


def set_font():
    for pattern in ("/usr/share/fonts/**/Inter*.otf", "/usr/share/fonts/**/Inter*.ttf",
                    "/Library/Fonts/Inter*.*", "~/Library/Fonts/Inter*.*", "C:/Windows/Fonts/Inter*.*"):
        for f in glob.glob(pattern, recursive=True):
            try:
                fm.fontManager.addfont(f)
            except Exception:
                pass
    names = {f.name for f in fm.fontManager.ttflist}
    plt.rcParams["font.family"] = "Inter" if "Inter" in names else "sans-serif"
