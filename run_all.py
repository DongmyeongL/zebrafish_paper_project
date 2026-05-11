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
]


def main():
    for script in SCRIPTS:
        path = ROOT / "figures" / script
        print(f"Running {path.relative_to(ROOT)}")
        subprocess.run([sys.executable, str(path)], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
