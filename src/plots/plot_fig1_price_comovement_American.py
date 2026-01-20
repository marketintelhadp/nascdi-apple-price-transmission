import os
import pandas as pd
import matplotlib.pyplot as plt

DATA = "data/model/weekly_prices_all_posneg.csv"
OUT = "results/nardl/plots"
os.makedirs(OUT, exist_ok=True)

VARIETY = "American"   # change to "Delicious" if needed

df = pd.read_csv(DATA, parse_dates=["week_end"])

df = df[df["variety"] == VARIETY]

markets = ["Shopian", "Sopore"]
grades = ["A", "B"]

fig, axes = plt.subplots(2, 2, figsize=(12, 6), sharex=True, sharey=False)

for i, m in enumerate(markets):
    for j, g in enumerate(grades):
        ax = axes[i, j]
        sub = df[
            (df["producer_market"] == m) &
            (df["grade"] == g)
        ].sort_values("week_end")

        ax.plot(sub["week_end"], sub["avg_price_prod"],
                label="Producer", color="tab:blue")
        ax.plot(sub["week_end"], sub["avg_price_term"],
                label="Terminal (Azadpur)", color="tab:red")

        ax.set_title(f"{m} | Grade {g}")
        if i == 1:
            ax.set_xlabel("Week")
        if j == 0:
            ax.set_ylabel("Price (₹/kg)")
        if i == 0 and j == 1:
            ax.legend(frameon=False)

plt.suptitle(
    f"Weekly Producer–Terminal Apple Price Co-movement ({VARIETY})",
    fontsize=14
)
plt.tight_layout(rect=[0, 0, 1, 0.95])

plt.savefig(
    os.path.join(OUT, "fig1_price_comovement_American.png"),
    dpi=300
)
plt.close()

print("✅ Figure 1 saved: fig1_price_comovement.png")
