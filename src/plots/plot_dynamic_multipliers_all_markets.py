import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

# ------------------
# CONFIG
# ------------------
IN_DIR = "data/model"
OUT_DIR = "results/nardl/plots/plots_advanced"
os.makedirs(OUT_DIR, exist_ok=True)

HORIZON = 24
MAX_LAGS_Y = 4
MAX_LAGS_X = 4
MIN_OBS = 60

# ------------------
# Helpers
# ------------------
def parse_name(fname):
    base = os.path.basename(fname).replace("_posneg.csv", "")
    m = re.match(r"weekly_model_(.+)_(American|Delicious)_(A|B)", base)
    if not m:
        return None
    return {
        "producer_market": m.group(1).replace("_", " "),
        "variety": m.group(2),
        "grade": m.group(3)
    }

def build_design(df):
    d = df.sort_values("week_end").copy()

    y="avg_price_term"; x="avg_price_prod"; p="NASCDI_pos"; n="NASCDI_neg"
    d["dy"]=d[y].diff(); d["dx"]=d[x].diff()
    d["dp"]=d[p].diff(); d["dn"]=d[n].diff()

    d["y_L1"]=d[y].shift(1); d["x_L1"]=d[x].shift(1)
    d["p_L1"]=d[p].shift(1); d["n_L1"]=d[n].shift(1)

    for L in range(1, MAX_LAGS_Y+1):
        d[f"dy_L{L}"]=d["dy"].shift(L)

    for L in range(0, MAX_LAGS_X+1):
        d[f"dx_L{L}"]=d["dx"].shift(L)
        d[f"dp_L{L}"]=d["dp"].shift(L)
        d[f"dn_L{L}"]=d["dn"].shift(L)

    cols = ["dy","y_L1","x_L1","p_L1","n_L1"] + \
           [f"dy_L{L}" for L in range(1,MAX_LAGS_Y+1)] + \
           [f"dx_L{L}" for L in range(0,MAX_LAGS_X+1)] + \
           [f"dp_L{L}" for L in range(0,MAX_LAGS_X+1)] + \
           [f"dn_L{L}" for L in range(0,MAX_LAGS_X+1)]

    d2 = d.dropna(subset=cols)
    Y = d2["dy"]
    X = add_constant(d2[[c for c in cols if c!="dy"]], has_constant="add")
    return d2, Y, X

def simulate_irf(params, shock="pos"):
    y_lag = x_lag = p_lag = n_lag = 0.0
    dy_hist = [0.0]*MAX_LAGS_Y
    dx_hist = [0.0]*(MAX_LAGS_X+1)
    dp_hist = [0.0]*(MAX_LAGS_X+1)
    dn_hist = [0.0]*(MAX_LAGS_X+1)

    out = []
    y = 0.0

    for t in range(HORIZON+1):
        dp0 = 1.0 if (shock=="pos" and t==0) else 0.0
        dn0 = 1.0 if (shock=="neg" and t==0) else 0.0

        dx_hist = [0.0] + dx_hist[:-1]
        dp_hist = [dp0] + dp_hist[:-1]
        dn_hist = [dn0] + dn_hist[:-1]

        dy = (
            params.get("const",0)
            + params.get("y_L1",0)*y_lag
            + params.get("x_L1",0)*x_lag
            + params.get("p_L1",0)*p_lag
            + params.get("n_L1",0)*n_lag
            + sum(params.get(f"dy_L{i}",0)*dy_hist[i-1] for i in range(1,MAX_LAGS_Y+1))
            + sum(params.get(f"dx_L{i}",0)*dx_hist[i] for i in range(0,MAX_LAGS_X+1))
            + sum(params.get(f"dp_L{i}",0)*dp_hist[i] for i in range(0,MAX_LAGS_X+1))
            + sum(params.get(f"dn_L{i}",0)*dn_hist[i] for i in range(0,MAX_LAGS_X+1))
        )

        y += dy
        y_lag = y
        p_lag += dp0
        n_lag += dn0
        dy_hist = [dy] + dy_hist[:-1]
        out.append(y)

    return np.array(out)

# ------------------
# MAIN
# ------------------
files = glob.glob(os.path.join(IN_DIR, "weekly_model_*_posneg.csv"))
series = []

for fp in files:
    meta = parse_name(fp)
    if not meta:
        continue

    df = pd.read_csv(fp)
    df["week_end"] = pd.to_datetime(df["week_end"], errors="coerce")
    df = df.dropna()

    d2, Y, X = build_design(df)
    if len(d2) < MIN_OBS:
        continue

    model = OLS(Y, X).fit()
    params = model.params.to_dict()

    for shock in ["pos","neg"]:
        path = simulate_irf(params, shock)
        for h, val in enumerate(path):
            series.append({
                "producer_market": meta["producer_market"],
                "variety": meta["variety"],
                "grade": meta["grade"],
                "shock": shock,
                "horizon": h,
                "response": val
            })

plot_df = pd.DataFrame(series)

# ------------------
# PLOT
# ------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)

layout = {
    ("American","pos"): (0,0),
    ("American","neg"): (0,1),
    ("Delicious","pos"): (1,0),
    ("Delicious","neg"): (1,1),
}

for (var, shock), (i,j) in layout.items():
    ax = axes[i,j]
    sub = plot_df[(plot_df["variety"]==var) & (plot_df["shock"]==shock)]

    for (m,g), grp in sub.groupby(["producer_market","grade"]):
        ax.plot(grp["horizon"], grp["response"], label=f"{m}–G{g}")

    ax.axhline(0, ls="--", lw=1)
    ax.set_title(f"{var} | NASCDI {'+' if shock=='pos' else '−'}")

axes[1,0].set_xlabel("Weeks")
axes[1,1].set_xlabel("Weeks")
axes[0,0].set_ylabel("Price response")
axes[1,0].set_ylabel("Price response")

handles, labels = axes[0,0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4)

fig.suptitle("Unified Dynamic Multipliers of NASCDI Shocks Across Markets", fontsize=14)
fig.tight_layout(rect=[0,0.05,1,0.95])

out = os.path.join(OUT_DIR, "fig_unified_dynamic_multipliers.png")
plt.savefig(out, dpi=300)
plt.close()

print("✅ Saved:", out)
