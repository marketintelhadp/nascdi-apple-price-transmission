import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/nardl/nardl_results_table.csv")

plt.figure(figsize=(6, 5))

plt.scatter(df["bounds_F"], df["ECT_y_L1"])
plt.axhline(0, color="black", linestyle="--")
plt.axvline(3.87, color="red", linestyle=":")

plt.xlabel("Bounds F-statistic")
plt.ylabel("ECT coefficient")
plt.title("Cointegration strength and speed of adjustment")
plt.tight_layout()
plt.savefig("results/nardl/plots/fig6_bounds_vs_ect.png", dpi=300)
plt.close()
