import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

DATA_PATH = "results/nardl/nardl_results_table.csv"
OUT_DIR = "results/plots"
os.makedirs(OUT_DIR, exist_ok=True)

# Load results
df = pd.read_csv(DATA_PATH)

# Safety check
required_cols = [
    "producer_market", "variety", "grade",
    "LR_effect_NASCDI_pos", "LR_effect_NASCDI_neg"
]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Create x-axis label
df["label"] = (
    df["producer_market"] + " | " +
    df["variety"] + " | G" +
    df["grade"].astype(str)
)

# Sort for clean plotting
df = df.sort_values(["producer_market", "variety", "grade"])

x = np.arange(len(df))
width = 0.35

plt.figure(figsize=(15, 6))

plt.bar(
    x - width / 2,
    df["LR_effect_NASCDI_pos"],
    width,
    label="Positive NASCDI (Disruptions)",
    color="#d73027"
)

plt.bar(
    x + width / 2,
    df["LR_effect_NASCDI_neg"],
    width,
    label="Negative NASCDI (Easing)",
    color="#4575b4"
)

plt.axhline(0, color="black", linewidth=0.8)

plt.xticks(x, df["label"], rotation=45, ha="right")
plt.ylabel("Long-run price effect")
plt.title("Asymmetric Long-run Effects of Supply-chain Disruptions (NASCDI)")
plt.legend()
plt.tight_layout()

out_path = os.path.join(OUT_DIR, "asymmetric_nascdi_effects.png")
plt.savefig(out_path, dpi=300)
plt.close()

print(f"✅ Saved: {out_path}")
