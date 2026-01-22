import numpy as np
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# PATHS
# --------------------------------------------------
BASE = Path(__file__).resolve().parents[2]
RES_PATH = BASE / "results" / "nardl" / "nardl_results_log.csv"
OUT_DIR = BASE / "results" / "dynamic_multipliers"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PATH = OUT_DIR / "dynamic_multipliers_posneg.csv"

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
MAX_HORIZON = 20  # weeks

# --------------------------------------------------
# LOAD NARDL RESULTS
# --------------------------------------------------
df = pd.read_csv(RES_PATH)

required = [
    "producer_market", "variety", "grade",
    "ECT_y_L1",
    "LR_effect_NASCDI_pos",
    "LR_effect_NASCDI_neg"
]

missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required NARDL columns: {missing}")

rows = []

# --------------------------------------------------
# BUILD DYNAMIC MULTIPLIERS
# --------------------------------------------------
for _, r in df.iterrows():

    phi = r["ECT_y_L1"]
    beta_pos = r["LR_effect_NASCDI_pos"]
    beta_neg = r["LR_effect_NASCDI_neg"]

    if pd.isna(phi) or phi >= 0:
        continue  # unstable or invalid ECT

    for h in range(MAX_HORIZON + 1):
        dm_pos = beta_pos * (1 - (1 + phi) ** h)
        dm_neg = beta_neg * (1 - (1 + phi) ** h)

        rows.append({
            "producer_market": r["producer_market"],
            "variety": r["variety"],
            "grade": r["grade"],
            "horizon": h,
            "dyn_mult_pos": dm_pos,
            "dyn_mult_neg": dm_neg
        })

dyn = pd.DataFrame(rows)
dyn.to_csv(OUT_PATH, index=False)

print(f"[SUCCESS] Dynamic multipliers saved at:\n{OUT_PATH}")
