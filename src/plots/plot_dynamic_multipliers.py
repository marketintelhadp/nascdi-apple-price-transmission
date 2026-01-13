import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Paths
# -----------------------------
RES_PATH = "results/nardl/nardl_results_table.csv"
OUT_DIR = "results/nardl/plots/dynamic_multipliers"
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------
# Load results
# -----------------------------
df = pd.read_csv(RES_PATH)

H = 20  # horizon in weeks

# -----------------------------
# Dynamic multiplier function
# -----------------------------
def dynamic_path(lr, ect, H=20):
    """
    ECM-based dynamic multiplier approximation
    """
    path = []
    cum = 0.0
    for h in range(H):
        cum += lr * ((1 + ect) ** h)
        path.append(cum)
    return np.array(path)

# -----------------------------
# Plot per series
# -----------------------------
for _, r in df.iterrows():
    pos = dynamic_path(r["LR_effect_NASCDI_pos"], r["ECT_y_L1"], H)
    neg = dynamic_path(r["LR_effect_NASCDI_neg"], r["ECT_y_L1"], H)

    plt.figure(figsize=(7, 4))
    plt.plot(
        pos,
        label="Disruption intensification (ΔNASCDI > 0)",
        color="red",
        lw=2
    )
    plt.plot(
        neg,
        label="Normalization / easing (ΔNASCDI < 0)",
        color="blue",
        lw=2
    )
    plt.axhline(0, linestyle="--", color="black", alpha=0.6)

    label = f"{r['producer_market']} | {r['variety']} | G{r['grade']}"
    plt.title(f"Dynamic multipliers: {label}")
    plt.xlabel("Weeks after shock")
    plt.ylabel("Cumulative terminal price response")
    plt.legend()

    fname = f"dyn_{r['producer_market']}_{r['variety']}_{r['grade']}.png"
    fname = fname.replace(" ", "_")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, fname), dpi=200)
    plt.close()
df["speed"] = df["ECT_y_L1"].abs()

df.sort_values("speed", ascending=False, inplace=True)

plt.figure(figsize=(8,4))
plt.barh(
    df["producer_market"] + " | " + df["variety"] + " G" + df["grade"].astype(str),
    df["speed"]
)
plt.title("Speed of adjustment to long-run equilibrium")
plt.xlabel("|ECT|")
plt.tight_layout()
plt.savefig("results/nardl/plots/speed_of_adjustment.png", dpi=200)
plt.close()

print("✅ Dynamic multiplier plots created successfully.")

sd = df["LR_effect_NASCDI_pos"].std()

plt.figure(figsize=(7,4))

for _, r in df.iterrows():
    path = dynamic_path(sd, r["ECT_y_L1"])
    plt.plot(path, alpha=0.4)

plt.title("Scenario: 1-SD disruption shock")
plt.xlabel("Weeks")
plt.ylabel("Terminal price response")
plt.tight_layout()
plt.savefig("results/nardl/plots/scenario_disruption_sd.png", dpi=200)
plt.close()

df["vulnerability"] = abs(df["LR_effect_NASCDI_pos"]) * abs(df["ECT_y_L1"])

