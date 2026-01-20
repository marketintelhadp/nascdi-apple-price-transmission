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

# Safe composite label
df["label"] = (
    df["producer_market"] + " | " +
    df["variety"] + " | G" + df["grade"].astype(str)
)

plot_df = df[[
    "label",
    "LR_effect_NASCDI_pos",
    "LR_effect_NASCDI_neg"
]].dropna()

# Sort by disruption intensification effect
plot_df = plot_df.sort_values("LR_effect_NASCDI_pos")

# ---------------- plot ----------------
fig, ax = plt.subplots(figsize=(9, 6))

y = np.arange(len(plot_df))

ax.scatter(
    plot_df["LR_effect_NASCDI_pos"],
    y,
    color="#b22222",
    s=60,
    label="Disruption intensification (NASCDI_pos)"
)

ax.scatter(
    plot_df["LR_effect_NASCDI_neg"],
    y,
    color="#1f77b4",
    s=60,
    label="Disruption easing (NASCDI_neg)"
)

# Connecting lines to emphasize asymmetry
for i in range(len(plot_df)):
    ax.plot(
        [plot_df.iloc[i]["LR_effect_NASCDI_pos"],
         plot_df.iloc[i]["LR_effect_NASCDI_neg"]],
        [y[i], y[i]],
        color="gray",
        alpha=0.4
    )

ax.axvline(0, color="black", lw=0.8)
ax.set_yticks(y)
ax.set_yticklabels(plot_df["label"])
ax.set_xlabel("Long-run price effect")
ax.set_title("Asymmetric long-run price response to supply-chain disruptions")

ax.legend(frameon=False)
plt.tight_layout()
plt.savefig(OUT / "fig_lr_asymmetry_dotplot.png", dpi=300)
plt.close()
