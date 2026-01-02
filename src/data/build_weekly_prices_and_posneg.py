import pandas as pd
import numpy as np
import os

PRICE_PATH = "data/clean/prices_long.csv"
NASCDI_PATH = "data/nascdi/nascdi_event_weekly.csv"
OUT_DIR = "data/model"

os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------
# Load price data
# -----------------------------
prices = pd.read_csv(PRICE_PATH, parse_dates=["date"])

# KEEP TRADING DAYS ONLY (CRITICAL)
prices = prices[prices["mask"] == 1].copy()

# -----------------------------
# Load NASCDI (weekly)
# -----------------------------
nascdi = pd.read_csv(NASCDI_PATH)

if "date" in nascdi.columns:
    nascdi["week_end"] = pd.to_datetime(nascdi["date"])
elif "week_end" in nascdi.columns:
    nascdi["week_end"] = pd.to_datetime(nascdi["week_end"])
else:
    raise ValueError("NASCDI file must contain 'date' or 'week_end'")

nascdi = nascdi[["week_end", "NASCDI"]].sort_values("week_end")

# -----------------------------
# GLOBAL POS / NEG DECOMPOSITION (CORRECT)
# -----------------------------
nascdi["d_nascdi"] = nascdi["NASCDI"].diff()

nascdi["NASCDI_pos"] = np.where(nascdi["d_nascdi"] > 0, nascdi["d_nascdi"], 0.0)
nascdi["NASCDI_neg"] = np.where(nascdi["d_nascdi"] < 0, -nascdi["d_nascdi"], 0.0)

nascdi["NASCDI_pos"] = nascdi["NASCDI_pos"].cumsum()
nascdi["NASCDI_neg"] = nascdi["NASCDI_neg"].cumsum()

# -----------------------------
# Weekly aggregation of prices
# -----------------------------
prices["week_end"] = prices["date"].dt.to_period("W").dt.end_time

weekly = (
    prices
    .groupby(
        ["market_role", "market_name", "variety", "grade", "week_end"],
        as_index=False
    )
    .agg(avg_price=("avg_pk", "mean"))
)

# -----------------------------
# Split producer & terminal
# -----------------------------
prod = weekly[weekly["market_role"] == "producer"]
term = weekly[weekly["market_role"] == "terminal"]

# -----------------------------
# Merge producer ↔ terminal
# -----------------------------
merged = prod.merge(
    term,
    on=["week_end", "variety", "grade"],
    suffixes=("_prod", "_term")
)

merged = merged.dropna(subset=["avg_price_prod", "avg_price_term"])

# -----------------------------
# Merge NASCDI (already pos/neg)
# -----------------------------
merged = merged.merge(
    nascdi[["week_end", "NASCDI", "NASCDI_pos", "NASCDI_neg"]],
    on="week_end",
    how="left"
)

merged = merged.dropna(subset=["NASCDI", "NASCDI_pos", "NASCDI_neg"])

# -----------------------------
# Save master file
# -----------------------------
merged.to_csv(
    os.path.join(OUT_DIR, "weekly_prices_all_posneg.csv"),
    index=False
)

# -----------------------------
# Save individual model files
# -----------------------------
for market in merged["market_name_prod"].unique():
    for v in merged["variety"].unique():
        for g in merged["grade"].unique():
            sub = merged[
                (merged["market_name_prod"] == market) &
                (merged["variety"] == v) &
                (merged["grade"] == g)
            ]

            if len(sub) < 30:
                continue

            fname = f"weekly_model_{market}_{v}_{g}_posneg.csv"

            sub[[
                "week_end",
                "avg_price_prod",
                "avg_price_term",
                "NASCDI",
                "NASCDI_pos",
                "NASCDI_neg"
            ]].to_csv(os.path.join(OUT_DIR, fname), index=False)

print("✅ Weekly price + NASCDI POS/NEG datasets created correctly.")
