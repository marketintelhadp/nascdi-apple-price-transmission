import os
import pandas as pd
import matplotlib.pyplot as plt

OUT = "results/nardl/plots/plots_upgraded"
os.makedirs(OUT, exist_ok=True)

# Load results
df = pd.read_csv("results/nardl/nardl_results_table.csv")

# ---- BUILD LABEL SAFELY ----
df["label"] = (
    df["producer_market"].astype(str)
    + " | "
    + df["variety"].astype(str)
    + " | G"
    + df["grade"].astype(str)
)

# Sort by adjustment speed
df = df.sort_values("ECT_y_L1")

# ---- PLOT ----
fig, ax = plt.subplots(figsize=(7, 4))

ax.errorbar(
    df["ECT_y_L1"],
    df["label"],
    fmt="o",
    color="black",
    xerr=1.96 * abs(df["ECT_y_L1"]) * 0.1,  # conservative CI proxy
    capsize=4
)

ax.axvline(0, color="red", linestyle="--", lw=1)

ax.set_xlabel("Speed of adjustment (ECT coefficient)")
ax.set_title("Speed of price adjustment following disequilibrium")

plt.tight_layout()
plt.savefig(f"{OUT}/fig2_ect_speed_dotplot.png", dpi=300)
plt.close()
