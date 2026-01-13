import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RES_PATH = "results/nardl/nardl_results_table.csv"
OUT_DIR = "results/nardl/plots_set2"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(RES_PATH)

# Label
df["series"] = (
    df["producer_market"].astype(str) + " | " +
    df["variety"].astype(str) + " | G" + df["grade"].astype(str)
)

# --- Asymmetry index: difference in long-run effects
# (If pos = disruption intensification, neg = easing)
df["ASYM_LR_diff"] = df["LR_effect_NASCDI_pos"] - df["LR_effect_NASCDI_neg"]
df["ASYM_LR_absdiff"] = (df["LR_effect_NASCDI_pos"] - df["LR_effect_NASCDI_neg"]).abs()

# 1) Heatmap: LR_effect_NASCDI_pos
pivot_pos = df.pivot(index="producer_market", columns=["variety", "grade"], values="LR_effect_NASCDI_pos")
plt.figure(figsize=(12, 4))
plt.imshow(pivot_pos.values, aspect="auto")
plt.xticks(range(pivot_pos.shape[1]), [f"{c[0]}-G{c[1]}" for c in pivot_pos.columns], rotation=45, ha="right")
plt.yticks(range(pivot_pos.shape[0]), pivot_pos.index)
plt.colorbar(label="LR effect (ΔNASCDI>0) = Disruption intensification")
plt.title("Heatmap: Long-run effect of disruption intensification (NASCDI_pos)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "heatmap_LR_NASCDI_pos.png"), dpi=200)
plt.close()

# 2) Heatmap: LR_effect_NASCDI_neg
pivot_neg = df.pivot(index="producer_market", columns=["variety", "grade"], values="LR_effect_NASCDI_neg")
plt.figure(figsize=(12, 4))
plt.imshow(pivot_neg.values, aspect="auto")
plt.xticks(range(pivot_neg.shape[1]), [f"{c[0]}-G{c[1]}" for c in pivot_neg.columns], rotation=45, ha="right")
plt.yticks(range(pivot_neg.shape[0]), pivot_neg.index)
plt.colorbar(label="LR effect (ΔNASCDI<0) = Normalization/easing")
plt.title("Heatmap: Long-run effect of normalization/easing (NASCDI_neg)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "heatmap_LR_NASCDI_neg.png"), dpi=200)
plt.close()

# 3) Bar plot: asymmetry absolute difference
d2 = df.sort_values("ASYM_LR_absdiff", ascending=False)
plt.figure(figsize=(14, 4))
plt.bar(d2["series"], d2["ASYM_LR_absdiff"])
plt.xticks(rotation=45, ha="right")
plt.ylabel("|LR_pos - LR_neg|")
plt.title("Asymmetry strength (absolute long-run difference)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "bar_asymmetry_absdiff.png"), dpi=200)
plt.close()

print("✅ Set 2A saved in:", OUT_DIR)
