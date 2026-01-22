import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

# --------------------
# CONFIG
# --------------------
IN_DIR = "data/model"
OUT_DIR = "results/nardl/plots/plots_advanced"
os.makedirs(OUT_DIR, exist_ok=True)

HORIZON = 24
MAX_LAGS_Y = 4
MAX_LAGS_X = 4
MIN_OBS = 60
SHOCK_TYPE = "pos"   # "pos" or "neg"

# --------------------
# Helpers
# --------------------
def parse_name(fp):
    base = os.path.basename(fp).replace("_posneg.csv", "")
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

    for L in range(1,MAX_LAGS_Y+1):
        d[f"dy_L{L}"]=d["dy"].shift(L)
    for L in range(0,MAX_LAGS_X+1):
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

def simulate_path(params, shock):
    y = y_lag = x_lag = p_lag = n_lag = 0.0
    dy_hist=[0]*MAX_LAGS_Y
    dx_hist=[0]*(MAX_LAGS_X+1)
    dp_hist=[0]*(MAX_LAGS_X+1)
    dn_hist=[0]*(MAX_LAGS_X+1)

    out=[]
    for t in range(HORIZON+1):
        dp0 = 1.0 if (shock=="pos" and t==0) else 0.0
        dn0 = 1.0 if (shock=="neg" and t==0) else 0.0

        dx_hist=[0]+dx_hist[:-1]
        dp_hist=[dp0]+dp_hist[:-1]
        dn_hist=[dn0]+dn_hist[:-1]

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

        y+=dy
        y_lag=y
        p_lag+=dp0
        n_lag+=dn0
        dy_hist=[dy]+dy_hist[:-1]
        out.append(y)
    return np.array(out)

# --------------------
# BUILD SURFACE DATA
# --------------------
rows=[]
files=sorted(glob.glob(os.path.join(IN_DIR,"weekly_model_*_posneg.csv")))

for fp in files:
    meta=parse_name(fp)
    if not meta: continue

    df=pd.read_csv(fp)
    df["week_end"]=pd.to_datetime(df["week_end"],errors="coerce")
    df=df.dropna()

    d2,Y,X=build_design(df)
    if len(d2)<MIN_OBS: continue

    model=OLS(Y,X).fit()
    path=simulate_path(model.params, SHOCK_TYPE)

    label=f"{meta['producer_market']} | {meta['variety']} | G{meta['grade']}"
    for h,val in enumerate(path):
        rows.append({"label":label,"horizon":h,"response":val})

surf_df=pd.DataFrame(rows)

# Encode categorical axis
labels=surf_df["label"].unique()
label_map={l:i for i,l in enumerate(labels)}
surf_df["y_index"]=surf_df["label"].map(label_map)

# Grid
X=surf_df["horizon"].values
Y=surf_df["y_index"].values
Z=surf_df["response"].values

# --------------------
# PLOT
# --------------------
fig=plt.figure(figsize=(14,8))
ax=fig.add_subplot(111,projection="3d")

surf=ax.plot_trisurf(
    X,Y,Z,
    cmap="viridis",
    linewidth=0.2,
    antialiased=True,
    alpha=0.95
)

ax.set_xlabel("Weeks after disruption shock")
ax.set_ylabel("Market × Variety × Grade")
ax.set_zlabel("Dynamic price response (₹/quintal)")

ax.set_yticks(list(label_map.values()))
ax.set_yticklabels(list(label_map.keys()), fontsize=8)

title = f"Integrated 3D Dynamic Response Surface ({'NASCDI+' if SHOCK_TYPE=='pos' else 'NASCDI−'})"
ax.set_title(title, pad=20)

fig.colorbar(surf, shrink=0.6, aspect=15, label="Price response magnitude")

plt.tight_layout()
out=os.path.join(OUT_DIR,f"fig_3d_dynamic_surface_{SHOCK_TYPE}.png")
plt.savefig(out,dpi=300)
plt.close()

print("✅ Saved:", out)
