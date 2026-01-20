import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA = Path("results/nardl/nardl_results_table.csv")
OUT = Path("results/nardl/plots_upgraded")
OUT.mkdir(exist_ok=True)

df = pd.read_csv(DATA)

df["label"] = (
    df["producer_market"] + " | " +
    df["variety"] + " | G" + df["grade"].astype(str)
)

df["asymmetry_strength"] = abs(
    df["LR_effect_NASCDI_pos"] - df["LR_effect_NASCDI_neg"]
)

df = df.sort_values("asymmetry_strength")

fig, ax = plt.subplots(figsize=(8, 6))

ax.hlines(
    y=df["label"],
    xmin=0,
    xmax=df["asymmetry_strength"],
    color="#555"
)
ax.plot(
    df["asymmetry_strength"],
    df["label"],
    "o",
    color="#b22222"
)

ax.set_xlabel("Absolute long-run asymmetry")
ax.set_title("Strength of asymmetric price transmission")

plt.tight_layout()
plt.savefig(OUT / "fig_asymmetry_strength_lollipop.png", dpi=300)
plt.close()
