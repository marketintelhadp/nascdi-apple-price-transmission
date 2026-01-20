import pandas as pd
import matplotlib.pyplot as plt
import os

OUT = "results/nardl/plots/plots_upgraded"
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv("results/nardl/nardl_results_table.csv")

# ---- LABEL ----
df["label"] = df["producer_market"] + " | " + df["variety"] + " | G" + df["grade"].astype(str)

df = df.sort_values("LR_effect_NASCDI_pos")

fig, ax = plt.subplots(figsize=(7, 4))

ax.scatter(df["LR_effect_NASCDI_pos"], df["label"], color="black")
ax.axvline(0, color="red", linestyle="--")

ax.set_xlabel("Long-run price response to disruption intensification (NASCDI⁺)")
ax.set_title("Asymmetric long-run effects: Disruption intensification")

plt.tight_layout()
plt.savefig(f"{OUT}/fig3a_lr_disruption_intensification.png", dpi=300)
plt.close()


df = pd.read_csv("results/nardl/nardl_results_table.csv")

df["label"] = df["producer_market"] + " | " + df["variety"] + " | G" + df["grade"].astype(str)
df = df.sort_values("LR_effect_NASCDI_neg")

fig, ax = plt.subplots(figsize=(7, 4))

ax.scatter(df["LR_effect_NASCDI_neg"], df["label"], color="black")
ax.axvline(0, color="red", linestyle="--")

ax.set_xlabel("Long-run price response to disruption easing (NASCDI⁻)")
ax.set_title("Asymmetric long-run effects: Disruption easing")

plt.tight_layout()
plt.savefig(f"{OUT}/fig3b_lr_disruption_easing.png", dpi=300)
plt.close()

