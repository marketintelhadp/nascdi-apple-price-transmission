import pandas as pd
import matplotlib.pyplot as plt
import os

PRICE_PATH = "data/model/weekly_prices_all_posneg.csv"
NASCDI_PATH = "data/nascdi/nascdi_event_weekly.csv"
OUT_DIR = "results/nardl/plots"

os.makedirs(OUT_DIR, exist_ok=True)

# ------------------------------
# Load weekly price data
# ------------------------------
prices = pd.read_csv(PRICE_PATH, parse_dates=["week_end"])

df = prices[
    (prices["producer_market"] == "Sopore") &
    (prices["variety"] == "Delicious") &
    (prices["grade"] == "A")
].sort_values("week_end")

df["vol_prod"] = df["avg_price_prod"].rolling(8).std()
df["vol_term"] = df["avg_price_term"].rolling(8).std()

# ------------------------------
# Load NASCDI
# ------------------------------
nascdi = pd.read_csv(NASCDI_PATH)

if "week_end" in nascdi.columns:
    nascdi["week_end"] = pd.to_datetime(nascdi["week_end"])
elif "date" in nascdi.columns:
    nascdi["week_end"] = pd.to_datetime(nascdi["date"])
else:
    raise ValueError("NASCDI file must contain 'date' or 'week_end'")

# Align NASCDI to price window
nascdi = nascdi[
    nascdi["week_end"].between(df["week_end"].min(), df["week_end"].max())
]

# ------------------------------
# Plot
# ------------------------------
fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

# Panel A: Prices
axes[0].plot(df["week_end"], df["avg_price_prod"], label="Producer price", lw=1.5)
axes[0].plot(df["week_end"], df["avg_price_term"], label="Terminal price", lw=1.5)
axes[0].set_ylabel("Price (₹/kg)")
axes[0].legend()
axes[0].set_title("Figure 1. Market context and disruption environment")

# Panel B: Volatility
axes[1].plot(df["week_end"], df["vol_prod"], label="Producer volatility", lw=1.2)
axes[1].plot(df["week_end"], df["vol_term"], label="Terminal volatility", lw=1.2)
axes[1].set_ylabel("Rolling SD")
axes[1].legend()

# Panel C: NASCDI
axes[2].plot(nascdi["week_end"], nascdi["NASCDI"], color="black", lw=1.2)
axes[2].set_ylabel("NASCDI")
axes[2].set_xlabel("Week")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "Figure1_Market_Context.png"), dpi=300)
plt.close()

print("✅ Figure 1 saved successfully.")
