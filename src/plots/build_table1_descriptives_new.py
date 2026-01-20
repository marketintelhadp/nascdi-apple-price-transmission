import pandas as pd
import os

DATA_PATH = "data/model/weekly_prices_all_posneg.csv"
OUT_DIR = "results/nardl/tables"
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(DATA_PATH, parse_dates=["week_end"])

# -----------------------------
# Create derived variables
# -----------------------------
df["price_spread"] = df["avg_price_term"] - df["avg_price_prod"]

# -----------------------------
# Descriptive statistics function
# -----------------------------
def describe(series):
    return pd.Series({
        "Mean": series.mean(),
        "Std. Dev.": series.std(),
        "Min": series.min(),
        "Max": series.max()
    })

# -----------------------------
# Panel A: Prices
# -----------------------------
panel_a = pd.DataFrame({
    "Producer price": describe(df["avg_price_prod"]),
    "Terminal price": describe(df["avg_price_term"]),
    "Price spread": describe(df["price_spread"])
})

# -----------------------------
# Panel B: NASCDI
# -----------------------------
panel_b = pd.DataFrame({
    "NASCDI": describe(df["NASCDI"]),
    "NASCDI_pos": describe(df["NASCDI_pos"]),
    "NASCDI_neg": describe(df["NASCDI_neg"])
})

# -----------------------------
# Panel C: Sample coverage
# -----------------------------
panel_c = pd.DataFrame({
    "Statistic": [
        "Observations",
        "Producer markets",
        "Varieties",
        "Grades",
        "Time span (weeks)"
    ],
    "Value": [
        len(df),
        df["producer_market"].nunique(),
        df["variety"].nunique(),
        df["grade"].nunique(),
        df["week_end"].nunique()
    ]
})

# -----------------------------
# Save tables
# -----------------------------
panel_a.to_csv(os.path.join(OUT_DIR, "Table1_PanelA_Prices.csv"))
panel_b.to_csv(os.path.join(OUT_DIR, "Table1_PanelB_NASCDI.csv"))
panel_c.to_csv(os.path.join(OUT_DIR, "Table1_PanelC_Sample.csv"))

print("✅ Table 1 (Panels A–C) created successfully.")
