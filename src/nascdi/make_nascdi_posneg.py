import os
import glob
import pandas as pd
import numpy as np

MODEL_DIR = "data/model"
OUT_DIR = "data/model_posneg"

os.makedirs(OUT_DIR, exist_ok=True)

def add_posneg(df: pd.DataFrame, col="NASCDI") -> pd.DataFrame:
    df = df.sort_values("week_end").copy()

    # Ensure numeric
    df[col] = pd.to_numeric(df[col], errors="coerce")

    # ΔNASCDI
    df["d_nascdi"] = df[col].diff()

    # Positive/negative changes
    df["d_nascdi_pos"] = df["d_nascdi"].clip(lower=0)
    df["d_nascdi_neg"] = df["d_nascdi"].clip(upper=0)

    # Partial sums
    df["NASCDI_pos"] = df["d_nascdi_pos"].cumsum()
    df["NASCDI_neg"] = df["d_nascdi_neg"].cumsum()

    return df

# Process all weekly model files
files = sorted(glob.glob(os.path.join(MODEL_DIR, "weekly_model_*.csv")))
if not files:
    raise FileNotFoundError(f"No weekly model files found in {MODEL_DIR}")

kept = 0
for fp in files:
    df = pd.read_csv(fp, parse_dates=["week_end"])

    # Basic guards
    if "NASCDI" not in df.columns:
        print(f"SKIP (no NASCDI): {os.path.basename(fp)}")
        continue

    df2 = add_posneg(df, col="NASCDI")

    # Drop first row (diff is NaN) + any missing core values
   # Drop only rows without prices or NASCDI
    df2 = df2.dropna(
    subset=["avg_price_prod", "avg_price_term", "NASCDI"]
)


    out_fp = os.path.join(OUT_DIR, os.path.basename(fp).replace(".csv", "_posneg.csv"))
    df2.to_csv(out_fp, index=False)
    kept += 1

# Also build a master file for convenience
master_fp = os.path.join(MODEL_DIR, "weekly_prices_all.csv")
if os.path.exists(master_fp):
    m = pd.read_csv(master_fp, parse_dates=["week_end"])
    m2 = add_posneg(m, col="NASCDI")
    m2.to_csv(os.path.join(OUT_DIR, "weekly_prices_all_posneg.csv"), index=False)

print(f"✅ Created POS/NEG datasets: {kept} files saved in {OUT_DIR}")
