import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

IN_DIR = "data/model"
OUT_DIR = "results/nardl/plots/plots_advanced"
os.makedirs(OUT_DIR, exist_ok=True)

MAX_LAGS_Y = 4
MAX_LAGS_X = 4
MIN_OBS = 60
HORIZON = 24  # weeks

def parse_name(fname: str):
    base = os.path.basename(fname).replace("_posneg.csv", "")
    m = re.match(r"weekly_model_(.+)_(American|Delicious)_(A|B)", base)
    if not m:
        return ("Unknown", "Unknown", "Unknown")
    return m.group(1).replace("_", " "), m.group(2), m.group(3)

def build_ardl_design(df):
    d = df.sort_values("week_end").copy()

    y = "avg_price_term"
    x = "avg_price_prod"
    p = "NASCDI_pos"
    n = "NASCDI_neg"

    d["dy"] = d[y].diff()
    d["dx"] = d[x].diff()
    d["dp"] = d[p].diff()
    d["dn"] = d[n].diff()

    d["y_L1"] = d[y].shift(1)
    d["x_L1"] = d[x].shift(1)
    d["p_L1"] = d[p].shift(1)
    d["n_L1"] = d[n].shift(1)

    for L in range(1, MAX_LAGS_Y + 1):
        d[f"dy_L{L}"] = d["dy"].shift(L)

    for L in range(0, MAX_LAGS_X + 1):
        d[f"dx_L{L}"] = d["dx"].shift(L)
        d[f"dp_L{L}"] = d["dp"].shift(L)
        d[f"dn_L{L}"] = d["dn"].shift(L)

    cols_needed = ["dy","y_L1","x_L1","p_L1","n_L1"] + \
        [f"dy_L{L}" for L in range(1, MAX_LAGS_Y+1)] + \
        [f"dx_L{L}" for L in range(0, MAX_LAGS_X+1)] + \
        [f"dp_L{L}" for L in range(0, MAX_LAGS_X+1)] + \
        [f"dn_L{L}" for L in range(0, MAX_LAGS_X+1)]

    d2 = d.dropna(subset=cols_needed).copy()

    Y = d2["dy"]
    Xcols = ["y_L1","x_L1","p_L1","n_L1"] + \
        [f"dy_L{L}" for L in range(1, MAX_LAGS_Y+1)] + \
        [f"dx_L{L}" for L in range(0, MAX_LAGS_X+1)] + \
        [f"dp_L{L}" for L in range(0, MAX_LAGS_X+1)] + \
        [f"dn_L{L}" for L in range(0, MAX_LAGS_X+1)]

    X = add_constant(d2[Xcols], has_constant="add")
    return d2, Y, X, Xcols

def simulate_irf(params, shock="pos", horizon=24):
    """
    Simulate response of y-level to a 1-unit impulse in dp_0 (pos) or dn_0 (neg).
    Uses the estimated Δy equation recursively.
    """
    # coefficients
    a = params.get("y_L1", 0.0)
    b = params.get("x_L1", 0.0)
    c = params.get("p_L1", 0.0)
    d = params.get("n_L1", 0.0)

    alpha = np.array([params.get(f"dy_L{i}", 0.0) for i in range(1, MAX_LAGS_Y+1)])

    bx = np.array([params.get(f"dx_L{i}", 0.0) for i in range(0, MAX_LAGS_X+1)])
    bp = np.array([params.get(f"dp_L{i}", 0.0) for i in range(0, MAX_LAGS_X+1)])
    bn = np.array([params.get(f"dn_L{i}", 0.0) for i in range(0, MAX_LAGS_X+1)])

    # Set baseline state: all zeros (interpretable as deviations from steady state)
    dy_hist = [0.0]*MAX_LAGS_Y
    dx_hist = [0.0]*(MAX_LAGS_X+1)
    dp_hist = [0.0]*(MAX_LAGS_X+1)
    dn_hist = [0.0]*(MAX_LAGS_X+1)

    # Level lags
    y_lag = 0.0
    x_lag = 0.0
    p_lag = 0.0
    n_lag = 0.0

    y_level = 0.0
    path_level = []
    path_cum = []

    for t in range(horizon+1):
        # impulse at t=0 in dp0 or dn0
        if t == 0:
            if shock == "pos":
                dp0 = 1.0
                dn0 = 0.0
            else:
                dp0 = 0.0
                dn0 = 1.0
        else:
            dp0 = 0.0
            dn0 = 0.0

        # update "current" histories (index 0 is contemporaneous)
        dx_hist = [0.0] + dx_hist[:-1]
        dp_hist = [dp0] + dp_hist[:-1]
        dn_hist = [dn0] + dn_hist[:-1]

        # Δy equation
        dy = (
            params.get("const", 0.0)
            + a*y_lag + b*x_lag + c*p_lag + d*n_lag
            + float(np.dot(alpha, np.array(dy_hist)))
            + float(np.dot(bx, np.array(dx_hist)))
            + float(np.dot(bp, np.array(dp_hist)))
            + float(np.dot(bn, np.array(dn_hist)))
        )

        # update levels: y_t = y_{t-1} + Δy_t ; p/n levels also integrate dp/dn
        y_level += dy
        p_lag += dp0
        n_lag += dn0

        # update y lag for next step
        y_lag = y_level

        # update dy history
        dy_hist = [dy] + dy_hist[:-1]

        path_level.append(y_level)
        path_cum.append(sum(path_level))  # cumulative sum of level response

    return np.array(path_level), np.array(path_cum)

def main():
    files = sorted(glob.glob(os.path.join(IN_DIR, "weekly_model_*_posneg.csv")))
    if not files:
        raise FileNotFoundError(f"No model files found in {IN_DIR}")

    for fp in files:
        market, variety, grade = parse_name(fp)
        df = pd.read_csv(fp)
        if "week_end" not in df.columns:
            # try date
            if "date" in df.columns:
                df["week_end"] = pd.to_datetime(df["date"])
            else:
                print(f"SKIP {os.path.basename(fp)} (no week_end/date)")
                continue
        df["week_end"] = pd.to_datetime(df["week_end"], errors="coerce")
        df = df.dropna(subset=["week_end","avg_price_prod","avg_price_term","NASCDI_pos","NASCDI_neg"])

        d2, Y, X, Xcols = build_ardl_design(df)
        if len(d2) < MIN_OBS:
            print(f"SKIP {os.path.basename(fp)} (too short after lags: {len(d2)})")
            continue

        model = OLS(Y, X).fit()
        params = model.params

        lvl_pos, cum_pos = simulate_irf(params, shock="pos", horizon=HORIZON)
        lvl_neg, cum_neg = simulate_irf(params, shock="neg", horizon=HORIZON)

        label = f"{market} | {variety} | G{grade}"

        # Plot
        fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharex=True)

        ax[0].plot(range(HORIZON+1), lvl_pos, label="NASCDI⁺ impulse (intensification)")
        ax[0].plot(range(HORIZON+1), lvl_neg, label="NASCDI⁻ impulse (easing)")
        ax[0].axhline(0, linestyle="--", linewidth=1)
        ax[0].set_title("Dynamic multiplier (level response)")
        ax[0].set_xlabel("Weeks")
        ax[0].set_ylabel("Δ Terminal price level")

        ax[1].plot(range(HORIZON+1), cum_pos, label="Cumulative NASCDI⁺")
        ax[1].plot(range(HORIZON+1), cum_neg, label="Cumulative NASCDI⁻")
        ax[1].axhline(0, linestyle="--", linewidth=1)
        ax[1].set_title("Cumulative multiplier")
        ax[1].set_xlabel("Weeks")
        ax[1].set_ylabel("Cumulative response")

        fig.suptitle(label)
        ax[0].legend()
        ax[1].legend()
        fig.tight_layout()

        safe_name = f"{market}_{variety}_{grade}".replace(" ", "_")
        out = os.path.join(OUT_DIR, f"cum_dynamic_multiplier_{safe_name}.png")
        fig.savefig(out, dpi=220)
        plt.close(fig)

        print("✅ Saved:", out)

if __name__ == "__main__":
    main()
