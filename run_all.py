"""Run the complete pipeline: generate -> clean -> EDA -> train -> score -> dashboard."""
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).parent / "src"
for step in ["generate_data.py", "preprocess.py", "eda.py", "train_model.py", "predict.py", "dashboard.py"]:
    print(f"\n=== {step} ===")
    subprocess.run([sys.executable, str(SRC / step)], check=True, cwd=SRC)
print("\nDone. Open outputs/dashboard.html in a browser.")
