"""Run the full pipeline in order. Add --scrape to re-download data from Basketball-Reference first."""
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "figures"))

steps = ["02_build_dataset.py", "03_minutes_cutoff.py", "04_collinearity.py", "05_model.py",
         "06_relative_weights.py", "07_eda.py",
         "figures/positive_contribution.py", "figures/impact_score_eda.py",
         "figures/minutes_cutoff.py", "figures/ols_summary.py"]
if "--scrape" in sys.argv:
    steps.insert(0, "01_scrape_bbref.py")

for step in steps:
    print(f"\n=== {step} ===")
    runpy.run_path(str(HERE / step), run_name="__main__")
