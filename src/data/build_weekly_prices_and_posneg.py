import os
import numpy as np
import pandas as pd

PRICE_PATH = "data/clean/prices_long.csv"
NASCDI_PATH = "data/nascdi/nascdi_event_weekly.csv"

OUT_DIR = "data/model"
os.makedirs(OUT_DIR, exist_ok=True)

MIN_WEEKS_TO_SAVE = 60  # ensures enough obs after lags

def to_week_end_sun_midnight(dt: pd.Series) -> pd.Series:
    """
    Convert any datetime to week ending Sunday (W-SUN) at 00:00:00.
    """
    dt = pd.to_datetime(dt, errors="coerce")
    # Create weekly period ending Sunday, then take end_time, then normalize to midnight
    we = dt.dt.to_period("W-SUN").dt.end_time.dt.normalize()
    return we

def build_nascdi_posneg(nascdi_weekly: pd.DataFrame) -> pd.DataFrame:
    """
    Build NASCDI_pos and NASCDI_neg from weekly NASCDI series:
    d = diff(NASCDI)
    pos = cum sum of max(d,0)
    neg = cum sum of max(-d,0)
    """
    t = nascdI_weekly = nascdI_weekly.copy()

    t = t.sort_values("week_end").drop_duplicates("week_end")
    t["d_nascdi"] = t["NASCDI"].diff()

    t["NASCDI_pos"] = np.where(t["d_nascdi"] > 0, t["d_nascdi"], 0.0)
    t["NASCDI_neg"] = np.where(t["d_nascdi"] < 0, -t["d_nascdi"], 0.0)

    t["NASCDI_pos"] = t["NASCDI_pos"].cumsum()
    t["NASCDI_neg"] = t["NASCDI_neg"].cumsum()

    return t[["week_end", "NASCDI", "d_nascdi", "NASCDI_pos", "NASCDI_neg"]]

def main():
    # -----------------------------
    # Load prices
    # -----------------------------
    prices = pd.read_csv(PRICE_PATH, parse_dates=["date"])
    # keep only trading days if mask exists
    if "mask" in prices.columns:
        prices = prices[prices["mask"] == 1].copy()

    # Week alignment: Sunday end-of-week at midnight
    prices["week_end"] = to_week_end_sun_midnight(prices["date"])

    # -----------------------------
    # Weekly aggregation by market-role/market/variety/grade
    # -----------------------------
    weekly = (
        prices
        .groupby(["market_role", "market_name", "variety", "grade", "week_end"], as_index=False)
        .agg(avg_price=("avg_pk", "mean"))
    )

    # -----------------------------
    # Split producer & terminal
    # -----------------------------
    prod = weekly[weekly["market_role"].str.lower() == "producer"].copy()
    term = weekly[weekly["market_role"].str.lower() == "terminal"].copy()

    # -----------------------------
    # Merge producer -> terminal by (week_end, variety, grade)
    # Allows multiple producer markets with same terminal
    # -----------------------------
    merged = prod.merge(
        term[["week_end", "variety", "grade", "avg_price", "market_name"]],
        on=["week_end", "variety", "grade"],
        how="inner",
        suffixes=("_prod", "_term")
    )

    # Rename for clarity
    merged = merged.rename(columns={
        "avg_price_prod": "avg_price_prod",
        "avg_price_term": "avg_price_term",
        "market_name_prod": "producer_market",
        "market_name_term": "terminal_market"
    })

    # If suffix renaming didn’t happen as expected:
    if "avg_price_prod" not in merged.columns and "avg_price_prod_prod" in merged.columns:
        merged = merged.rename(columns={"avg_price_prod_prod": "avg_price_prod"})
    if "avg_price_term" not in merged.columns and "avg_price_term" not in merged.columns:
        # in the above merge, term avg_price comes as "avg_price"
        if "avg_price" in merged.columns:
            merged = merged.rename(columns={"avg_price": "avg_price_term"})

    # Clean: drop missing prices
    merged = merged.dropna(subset=["avg_price_prod", "avg_price_term"]).copy()

    # -----------------------------
    # Load NASCDI weekly and align week_end
    # -----------------------------
    nas = pd.read_csv(NASCDI_PATH)
    if "week_end" in nas.columns:
        nas["week_end"] = pd.to_datetime(nas["week_end"], errors="coerce")
    elif "date" in nas.columns:
        nas["week_end"] = pd.to_datetime(nas["date"], errors="coerce")
    else:
        raise ValueError("NASCDI file must contain 'date' or 'week_end'")

    nas["week_end"] = to_week_end_sun_midnight(nas["week_end"])

    if "NASCDI" not in nas.columns:
        raise ValueError("NASCDI file must contain a 'NASCDI' column")

    nas = nas[["week_end", "NASCDI"]].dropna().drop_duplicates("week_end").sort_values("week_end")

    # Build POS/NEG from NASCDI itself (correct method)
    nas_posneg = build_nascdi_posneg(nas)

    # Merge into price data
    merged = merged.merge(nas_posneg, on="week_end", how="left")
    merged = merged.dropna(subset=["NASCDI", "NASCDI_pos", "NASCDI_neg"]).copy()

    # -----------------------------
    # Save master file
    # -----------------------------
    master_path = os.path.join(OUT_DIR, "weekly_prices_all_posneg.csv")
    merged.to_csv(master_path, index=False)
    print(f"✅ Saved master: {master_path} | rows={len(merged)}")

    # -----------------------------
    # Save per-producer-market/variety/grade datasets
    # -----------------------------
    for (pm, v, g), sub in merged.groupby(["producer_market", "variety", "grade"]):
        sub = sub.sort_values("week_end")
        if len(sub) < MIN_WEEKS_TO_SAVE:
            # don’t write empty/short series
            continue

        fname = f"weekly_model_{pm}_{v}_{g}_posneg.csv"
        # sanitize filename
        fname = fname.replace(" ", "_")

        out_fp = os.path.join(OUT_DIR, fname)
        sub[[
            "week_end",
            "avg_price_prod",
            "avg_price_term",
            "NASCDI",
            "NASCDI_pos",
            "NASCDI_neg",
        ]].to_csv(out_fp, index=False)

    print("✅ Per-series model files saved (only non-empty, sufficient length).")

if __name__ == "__main__":
    main()
