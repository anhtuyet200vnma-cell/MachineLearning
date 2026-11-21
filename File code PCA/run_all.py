"""Run every script in order and store the console output in outputs/logs/."""
import subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOGS = HERE / "outputs" / "logs"
LOGS.mkdir(parents=True, exist_ok=True)
SCRIPTS = ["minimal_pipeline", "casestudy", "pca_experiment", "phu_luc_A_pca_visualization", "phu_luc_B_cross_validation",
           "phu_luc_C_bootstrap", "phu_luc_D_repeated_splits", "phu_luc_E_geometry", "phu_luc_F_extensions"]

for name in SCRIPTS:
    print(f"running {name}.py ...", flush=True)
    res = subprocess.run([sys.executable, str(HERE / f"{name}.py")], cwd=HERE, capture_output=True, text=True)
    (LOGS / f"{name}.log").write_text(res.stdout + res.stderr, encoding="utf-8")
    if res.returncode != 0:
        print(res.stderr); sys.exit(f"{name}.py failed")
print("all scripts finished; logs in outputs/logs/")
