# ============================================================
# Integrated 3D Dynamic Response Surface (NASCDI Intensification)
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.mplot3d import Axes3D  # noqa
from pathlib import Path

# ------------------------------------------------------------
# PATHS (ROBUST)
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]   # project root
DATA_PATH = BASE_DIR / "results" / "nardl" / "nardl_results_log.csv"
OUT_PATH = BASE_DIR / "plots_upgraded" / "fig_integrated_3d_dynamic_surface.png"

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------
df = pd.read_csv(DATA_PATH)

# ------------------------------------------------------------
# REQUIRED COLUMNS
# ------------------------------------------------------------
required_cols = [
    "producer_market",
    "variety",
    "grade",
    "horizon",
    "dyn_mult_pos"
]

missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# ------------------------------------------------------------
# LABEL CONSTRUCTION
# ------------------------------------------------------------
df["label"] = (
    df["producer_market"].astype(str) + " | " +
    df["variety"].astype(str) + " | " +
    df["grade"].astype(str)
)

labels = df["label"].unique()
horizons = np.sort(df["horizon"].unique())

# ------------------------------------------------------------
# BUILD 2D RESPONSE MATRIX (KEY FIX)
# ------------------------------------------------------------
Z = np.zeros((len(labels), len(horizons)))

for i, lab in enumerate(labels):
    sub = df[df["label"] == lab].sort_values("horizon")

    if len(sub) != len(horizons):
        raise ValueError(f"Incomplete horizon data for {lab}")

    Z[i, :] = sub["dyn_mult_pos"].values

X, Y = np.meshgrid(horizons, np.arange(len(labels)))

# ------------------------------------------------------------
# COLOUR NORMALIZATION (ZERO-CENTERED)
# ------------------------------------------------------------
norm = TwoSlopeNorm(
    vmin=np.min(Z),
    vcenter=0.0,
    vmax=np.max(Z)
)

# ------------------------------------------------------------
# PLOT
# ------------------------------------------------------------
fig = plt.figure(figsize=(17, 11))
ax = fig.add_subplot(111, projection="3d")

surf = ax.plot_surface(
    X, Y, Z,
    cmap="turbo",          # more colourful than coolwarm
    norm=norm,
    linewidth=0,
    antialiased=True,
    alpha=0.95
)

# Zero reference plane
ax.plot_surface(
    X, Y, np.zeros_like(Z),
    color="grey",
    alpha=0.15
)

# ------------------------------------------------------------
# AXES & LABELS (REVIEWER-READY)
# ------------------------------------------------------------
ax.set_xlabel("Weeks after NASCDI intensification", labelpad=14)
ax.set_ylabel("Market | Variety | Grade", labelpad=14)
ax.set_zlabel("Dynamic price response (₹/quintal)", labelpad=14)

ax.set_yticks(np.arange(len(labels)))
ax.set_yticklabels(labels, fontsize=8)

ax.set_title(
    "Integrated 3D Dynamic Price Response Surface to Supply-Chain Disruptions",
    fontsize=15,
    pad=20
)

# ------------------------------------------------------------
# COLORBAR
# ------------------------------------------------------------
cbar = fig.colorbar(
    surf,
    shrink=0.6,
    aspect=18,
    pad=0.08
)
cbar.set_label(
    "Price response magnitude\n(NASCDI intensification)",
    fontsize=11
)

# ------------------------------------------------------------
# VIEW ANGLE (OPTIMIZED FOR INTERPRETATION)
# ------------------------------------------------------------
ax.view_init(elev=28, azim=-135)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
plt.show()

print(f"[SUCCESS] Figure saved at: {OUT_PATH}")
