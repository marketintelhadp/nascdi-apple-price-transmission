import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

IN_DIR = "data/model"
OUT_DIR = "results/nardl/plots/plots_advanced"
os.makedirs(OUT_DIR, exist_ok=True)

MAX_LAGS_Y = 4
MAX_LAGS_X = 4
MIN_OBS = 60
HORIZON = 20
SHOCK_GRID = np.linspace(-4, 5, 20) # negative=easing, positive=intensification

def parse_name(fname: str):
    base = os.path.basename(fname).replace("_posneg.csv", "")
    m = re.match(r"weekly_model_(.+)_(American|Delicious)_(A|B)", base)
    if not m:
        return ("Unknown", "Unknown", "Unknown")
    return m.group(1).replace("_", " "), m.group(2), m.group(3)

def build_ardl_design(df):
    d = df.sort_values("week_end").copy()
    y="avg_price_term"; x="avg_price_prod"; p="NASCDI_pos"; n="NASCDI_neg"
    d["dy"]=d[y].diff(); d["dx"]=d[x].diff(); d["dp"]=d[p].diff(); d["dn"]=d[n].diff()
    d["y_L1"]=d[y].shift(1); d["x_L1"]=d[x].shift(1); d["p_L1"]=d[p].shift(1); d["n_L1"]=d[n].shift(1)

    for L in range(1, MAX_LAGS_Y+1): d[f"dy_L{L}"]=d["dy"].shift(L)
    for L in range(0, MAX_LAGS_X+1):
        d[f"dx_L{L}"]=d["dx"].shift(L)
        d[f"dp_L{L}"]=d["dp"].shift(L)
        d[f"dn_L{L}"]=d["dn"].shift(L)

    cols=["dy","y_L1","x_L1","p_L1","n_L1"] + \
        [f"dy_L{L}" for L in range(1,MAX_LAGS_Y+1)] + \
        [f"dx_L{L}" for L in range(0,MAX_LAGS_X+1)] + \
        [f"dp_L{L}" for L in range(0,MAX_LAGS_X+1)] + \
        [f"dn_L{L}" for L in range(0,MAX_LAGS_X+1)]
    d2=d.dropna(subset=cols).copy()

    Y=d2["dy"]
    Xcols=["y_L1","x_L1","p_L1","n_L1"] + \
        [f"dy_L{L}" for L in range(1,MAX_LAGS_Y+1)] + \
        [f"dx_L{L}" for L in range(0,MAX_LAGS_X+1)] + \
        [f"dp_L{L}" for L in range(0,MAX_LAGS_X+1)] + \
        [f"dn_L{L}" for L in range(0,MAX_LAGS_X+1)]
    X=add_constant(d2[Xcols], has_constant="add")
    return d2, Y, X

def simulate_path(params, shock_size, horizon):
    # shock_size>0 -> dp0 = shock_size ; shock_size<0 -> dn0 = |shock_size|
    dp0 = max(shock_size, 0.0)
    dn0 = max(-shock_size, 0.0)

    a=params.get("y_L1",0.0); b=params.get("x_L1",0.0); c=params.get("p_L1",0.0); d=params.get("n_L1",0.0)
    alpha=np.array([params.get(f"dy_L{i}",0.0) for i in range(1,MAX_LAGS_Y+1)])
    bx=np.array([params.get(f"dx_L{i}",0.0) for i in range(0,MAX_LAGS_X+1)])
    bp=np.array([params.get(f"dp_L{i}",0.0) for i in range(0,MAX_LAGS_X+1)])
    bn=np.array([params.get(f"dn_L{i}",0.0) for i in range(0,MAX_LAGS_X+1)])

    dy_hist=[0.0]*MAX_LAGS_Y
    dx_hist=[0.0]*(MAX_LAGS_X+1)
    dp_hist=[0.0]*(MAX_LAGS_X+1)
    dn_hist=[0.0]*(MAX_LAGS_X+1)

    y_lag=x_lag=p_lag=n_lag=0.0
    y_level=0.0
    out=[]

    for t in range(horizon+1):
        if t==0:
            dp = dp0; dn = dn0
        else:
            dp = 0.0; dn = 0.0

        dx_hist=[0.0]+dx_hist[:-1]
        dp_hist=[dp]+dp_hist[:-1]
        dn_hist=[dn]+dn_hist[:-1]

        dy=(params.get("const",0.0)
            +a*y_lag+b*x_lag+c*p_lag+d*n_lag
            +float(np.dot(alpha,np.array(dy_hist)))
            +float(np.dot(bx,np.array(dx_hist)))
            +float(np.dot(bp,np.array(dp_hist)))
            +float(np.dot(bn,np.array(dn_hist)))
        )

        y_level += dy
        p_lag += dp
        n_lag += dn
        y_lag = y_level
        dy_hist=[dy]+dy_hist[:-1]
        out.append(y_level)
    return np.array(out)

def main():
    files = sorted(glob.glob(os.path.join(IN_DIR, "weekly_model_*_posneg.csv")))
    if not files:
        raise FileNotFoundError("No model files found.")

    for fp in files:
        market, variety, grade = parse_name(fp)
        df = pd.read_csv(fp)

        if "week_end" not in df.columns:
            if "date" in df.columns:
                df["week_end"] = pd.to_datetime(df["date"])
            else:
                continue
        df["week_end"] = pd.to_datetime(df["week_end"], errors="coerce")
        df = df.dropna(subset=["week_end","avg_price_prod","avg_price_term","NASCDI_pos","NASCDI_neg"])

        d2, Y, X = build_ardl_design(df)
        if len(d2) < MIN_OBS:
            continue

        model = OLS(Y, X).fit()
        params = dict(zip(model.params.index, model.params.values))

        H = np.arange(HORIZON+1)
        S = SHOCK_GRID
        Z = np.zeros((len(S), len(H)))

        for i, s in enumerate(S):
            Z[i, :] = simulate_path(params, shock_size=s, horizon=HORIZON)

        HH, SS = np.meshgrid(H, S)

        fig = plt.figure(figsize=(11, 5))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(HH, SS, Z, linewidth=0, antialiased=True, alpha=0.95)

        ax.set_xlabel("Horizon (weeks)")
        ax.set_ylabel("Shock size (NASCDI: + intensification, − easing)")
        ax.set_zlabel("Terminal price response (level)")
        ax.set_title(f"3D adjustment surface — {market} | {variety} | G{grade}")

        safe = f"{market}_{variety}_{grade}".replace(" ", "_")
        out = os.path.join(OUT_DIR, f"surface3d_{safe}.png")
        plt.tight_layout()
        plt.savefig(out, dpi=220)
        plt.close(fig)
        print("✅ Saved:", out)

if __name__ == "__main__":
    main()
