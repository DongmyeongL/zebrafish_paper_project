import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SCRIPTS = [
    "figure9_clean.py",
    "figure12_clean.py",
    "figure13_clean.py",
    "figure14_clean.py",
    "figure14_celegans_full_combined_FINAL.py",
    "figure15_drosophila_full_combined_FINAL.py",
    "figure_fc_dynamics_final_fc_shift_summary.py",
    "figure_sc_fc_final_overview.py",
    "figure_supply_0.py",
    "figure_supply_1.py",
    "figure_supply_2.py",
    "figure_supply_5.py",
    "figure_supply_10.py",
    "figure_supply_11.py",
    "figure_supply_13.py",
    "figure_supply_14.py",
    "figure_supply_15.py",
    "figure_supply_16.py",
]


def main():
    for script in SCRIPTS:
        path = ROOT / "figures" / script
        print(f"Running {path.relative_to(ROOT)}")
        subprocess.run([sys.executable, str(path)], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
