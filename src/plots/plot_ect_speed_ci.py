import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ---------------- paths ----------------
DATA = Path("results/nardl/nardl_results_table.csv")
OUT = Path("results/nardl/plots_upgraded")
OUT.mkdir(parents=True, exist_ok=True)

# ---------------- load ----------------
df = pd.read_csv(DATA)

df["label"] = (
    df["producer_market"] + " | " +
    df["variety"] + " | G" + df["grade"].astype(str)
)

plot_df = df[["label", "ECT_y_L1"]].dropna()
plot_df = plot_df.sort_values("ECT_y_L1")

# ---------------- plot ----------------
fig, ax = plt.subplots(figsize=(8, 6))

y = np.arange(len(plot_df))

ax.scatter(
    plot_df["ECT_y_L1"],
    y,
    s=70,
    color="#2ca02c"
)

ax.axvline(0, color="black", lw=0.8)
ax.axvline(-0.2, color="gray", lw=0.6, linestyle="--", alpha=0.6)

ax.set_yticks(y)
ax.set_yticklabels(plot_df["label"])
ax.set_xlabel("Error-correction coefficient (speed of adjustment)")
ax.set_title("Speed of price adjustment toward long-run equilibrium")

plt.tight_layout()
plt.savefig(OUT / "fig_ect_speed_dotplot.png", dpi=300)
plt.close()
