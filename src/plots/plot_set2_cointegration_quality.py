import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RES_PATH = "results/nardl/nardl_results_table.csv"
DIAG_PATH = "results/nardl/nardl_diagnostics.csv"
OUT_DIR = "results/nardl/plots_set2"
os.makedirs(OUT_DIR, exist_ok=True)

res = pd.read_csv(RES_PATH)
diag = pd.read_csv(DIAG_PATH)

df = res.merge(diag, on=["producer_market", "variety", "grade"], how="left")

df["series"] = (
    df["producer_market"].astype(str) + " | " +
    df["variety"].astype(str) + " | G" + df["grade"].astype(str)
)

# 1) ECT plot (speed of adjustment)
# ECT_y_L1 should be negative and significant for stable adjustment
d1 = df.sort_values("ECT_y_L1")
plt.figure(figsize=(14, 4))
plt.bar(d1["series"], d1["ECT_y_L1"])
plt.axhline(0, linestyle="--")
plt.xticks(rotation=45, ha="right")
plt.ylabel("ECT (y_{t-1} coefficient)")
plt.title("Error-correction term (ECT): expected negative for adjustment")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "bar_ECT_yL1.png"), dpi=200)
plt.close()

# 2) Cointegration “evidence”: bounds_F vs p-value scatter
plt.figure(figsize=(6, 4))
plt.scatter(df["bounds_F"], df["bounds_F_pvalue"])
plt.axhline(0.05, linestyle="--")
plt.xlabel("Bounds-like F-stat")
plt.ylabel("p-value (levels block)")
plt.title("Cointegration evidence (pragmatic bounds-like test)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "scatter_boundsF_pvalue.png"), dpi=200)
plt.close()

# 3) Model fit comparison (R2)
d3 = df.sort_values("R2", ascending=False)
plt.figure(figsize=(14, 4))
plt.bar(d3["series"], d3["R2"])
plt.xticks(rotation=45, ha="right")
plt.ylabel("R²")
plt.title("Model fit (R²) across series")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "bar_R2.png"), dpi=200)
plt.close()

# 4) Residual diagnostics summary: Ljung-Box + BP p-values
# Higher p-values are generally “better” (fail to reject)
plt.figure(figsize=(6, 4))
plt.scatter(df["ljungbox_p(8)"], df["bp_heterosk_p"])
plt.axvline(0.05, linestyle="--")
plt.axhline(0.05, linestyle="--")
plt.xlabel("Ljung-Box p(8) (no autocorr if >0.05)")
plt.ylabel("Breusch-Pagan p (homosked if >0.05)")
plt.title("Residual diagnostics map")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "scatter_residual_diagnostics.png"), dpi=200)
plt.close()

print("✅ Set 2B saved in:", OUT_DIR)
