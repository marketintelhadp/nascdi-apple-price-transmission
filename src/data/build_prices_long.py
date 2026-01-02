import pandas as pd
import numpy as np
import glob
import os

RAW_DIR = "data/raw/prices"
OUT_DIR = "data/clean"
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------
# Utilities
# -----------------------------
def robust_parse_date(s):
    d = pd.to_datetime(s, errors="coerce")
    mask = d.isna()
    if mask.any():
        d.loc[mask] = pd.to_datetime(s[mask], errors="coerce", dayfirst=True)
    return d

def fix_prices(df):
    for c in ["Min Price (per kg)", "Max Price (per kg)", "Avg Price (per kg)"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df.loc[df[c] <= 0, c] = np.nan

    # Swap Min > Max
    bad = df["Min Price (per kg)"] > df["Max Price (per kg)"]
    df.loc[bad, ["Min Price (per kg)", "Max Price (per kg)"]] = \
        df.loc[bad, ["Max Price (per kg)", "Min Price (per kg)"]].values

    # Recompute Avg if missing but Min & Max exist (Mask==1)
    cond = (
        df["Mask"].eq(1) &
        df["Avg Price (per kg)"].isna() &
        df["Min Price (per kg)"].notna() &
        df["Max Price (per kg)"].notna()
    )
    df.loc[cond, "Avg Price (per kg)"] = (
        df.loc[cond, "Min Price (per kg)"] +
        df.loc[cond, "Max Price (per kg)"]
    ) / 2

    return df

def normalize_market(district, market):
    district = str(district).strip().lower() if pd.notna(district) else ""
    market = str(market).strip().lower() if pd.notna(market) else ""

    if district.startswith("sopore"):
        return "Sopore"
    if district.startswith("shopian"):
        return "Shopian"
    if district.startswith("azadpur"):
        return "Azadpur"

    # fallback: infer from market field
    if "azadpur" in market:
        return "Azadpur"

    return district.title() if district else "Unknown"


def normalize_variety(v):
    v = str(v).strip().lower()
    if "delicious" in v:
        return "Delicious"
    if "american" in v:
        return "American"
    return None

def normalize_grade(g):
    g = str(g).strip().upper()
    if g in ["A", "B"]:
        return g
    return None

# -----------------------------
# Main ingestion
# -----------------------------
frames = []

for fp in glob.glob(os.path.join(RAW_DIR, "*.csv")):
    df = pd.read_csv(fp)

    # Standardize date
    df["date"] = robust_parse_date(df["Date"])

    # Fix corrupted district entry
    df["District"] = df["District"].replace("[", "Shopian")

    # Market standardization
    df["District"] = df["District"].astype(str).str.strip()
    df["Market"] = df["Market"].astype(str).str.strip()

    df["market_name"] = df.apply(
        lambda r: normalize_market(r["District"], r["Market"]),
        axis=1
    )

    df["market_role"] = np.where(
        df["market_name"].eq("Azadpur"),
        "terminal",
        "producer"
    )

    df["market_subnode"] = df["Market"].astype(str).str.strip()

    # Normalize product identifiers
    df["variety"] = df["Variety"].apply(normalize_variety)
    df["grade"] = df["Grade"].apply(normalize_grade)

    # Drop irrelevant rows
    df = df[df["variety"].notna() & df["grade"].notna()]
    df = df[df["date"].notna()]

    # Price cleaning
    df = fix_prices(df)

    # Select and rename
    out = df[[
        "date",
        "market_role",
        "market_name",
        "market_subnode",
        "variety",
        "grade",
        "Min Price (per kg)",
        "Max Price (per kg)",
        "Avg Price (per kg)",
        "Mask"
    ]].rename(columns={
        "Min Price (per kg)": "min_pk",
        "Max Price (per kg)": "max_pk",
        "Avg Price (per kg)": "avg_pk",
        "Mask": "mask"
    })

    out["source_file"] = os.path.basename(fp)
    frames.append(out)

# Combine all files
prices_long = pd.concat(frames, ignore_index=True)

# Sort
prices_long = prices_long.sort_values(
    ["market_name", "variety", "grade", "date"]
)

# Save
OUT_PATH = os.path.join(OUT_DIR, "prices_long.csv")
prices_long.to_csv(OUT_PATH, index=False)

print("✅ prices_long.csv created successfully")
print("Rows:", len(prices_long))
print("Markets:", prices_long["market_name"].unique())
print("Varieties:", prices_long["variety"].unique())
print("Grades:", prices_long["grade"].unique())
