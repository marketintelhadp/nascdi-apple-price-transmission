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
HORIZON = 24
N_DRAWS = 500  # increase to 1000 for smoother bands

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
    return d2, Y, X

def simulate_path(params, shock="pos", horizon=24):
    a = params.get("y_L1", 0.0)
    b = params.get("x_L1", 0.0)
    c = params.get("p_L1", 0.0)
    d = params.get("n_L1", 0.0)

    alpha = np.array([params.get(f"dy_L{i}", 0.0) for i in range(1, MAX_LAGS_Y+1)])
    bx = np.array([params.get(f"dx_L{i}", 0.0) for i in range(0, MAX_LAGS_X+1)])
    bp = np.array([params.get(f"dp_L{i}", 0.0) for i in range(0, MAX_LAGS_X+1)])
    bn = np.array([params.get(f"dn_L{i}", 0.0) for i in range(0, MAX_LAGS_X+1)])

    dy_hist = [0.0]*MAX_LAGS_Y
    dx_hist = [0.0]*(MAX_LAGS_X+1)
    dp_hist = [0.0]*(MAX_LAGS_X+1)
    dn_hist = [0.0]*(MAX_LAGS_X+1)

    y_lag = x_lag = p_lag = n_lag = 0.0
    y_level = 0.0
    path = []

    for t in range(horizon+1):
        if t == 0:
            dp0 = 1.0 if shock == "pos" else 0.0
            dn0 = 1.0 if shock == "neg" else 0.0
        else:
            dp0 = 0.0
            dn0 = 0.0

        dx_hist = [0.0] + dx_hist[:-1]
        dp_hist = [dp0] + dp_hist[:-1]
        dn_hist = [dn0] + dn_hist[:-1]

        dy = (
            params.get("const", 0.0)
            + a*y_lag + b*x_lag + c*p_lag + d*n_lag
            + float(np.dot(alpha, np.array(dy_hist)))
            + float(np.dot(bx, np.array(dx_hist)))
            + float(np.dot(bp, np.array(dp_hist)))
            + float(np.dot(bn, np.array(dn_hist)))
        )

        y_level += dy
        p_lag += dp0
        n_lag += dn0
        y_lag = y_level
        dy_hist = [dy] + dy_hist[:-1]

        path.append(y_level)

    return np.array(path)

def main():
    files = sorted(glob.glob(os.path.join(IN_DIR, "weekly_model_*_posneg.csv")))
    if not files:
        raise FileNotFoundError(f"No model files found in {IN_DIR}")

    for fp in files:
        market, variety, grade = parse_name(fp)
        df = pd.read_csv(fp)

        if "week_end" not in df.columns:
            if "date" in df.columns:
                df["week_end"] = pd.to_datetime(df["date"])
            else:
                print("SKIP (no week_end/date):", os.path.basename(fp))
                continue

        df["week_end"] = pd.to_datetime(df["week_end"], errors="coerce")
        df = df.dropna(subset=["week_end","avg_price_prod","avg_price_term","NASCDI_pos","NASCDI_neg"])

        d2, Y, X = build_ardl_design(df)
        if len(d2) < MIN_OBS:
            print(f"SKIP {os.path.basename(fp)} (too short: {len(d2)})")
            continue

        model = OLS(Y, X).fit()
        beta = model.params.values
        cov = model.cov_params().values
        names = model.params.index.tolist()

        # point estimate
        params_hat = dict(zip(names, model.params.values))
        irf_pos_hat = simulate_path(params_hat, "pos", HORIZON)
        irf_neg_hat = simulate_path(params_hat, "neg", HORIZON)

        # bootstrap bands
        draws = np.random.multivariate_normal(beta, cov, size=N_DRAWS)

        irf_pos_draws = np.zeros((N_DRAWS, HORIZON+1))
        irf_neg_draws = np.zeros((N_DRAWS, HORIZON+1))

        for i in range(N_DRAWS):
            params_i = dict(zip(names, draws[i]))
            irf_pos_draws[i] = simulate_path(params_i, "pos", HORIZON)
            irf_neg_draws[i] = simulate_path(params_i, "neg", HORIZON)

        def band(x):
            lo = np.quantile(x, 0.025, axis=0)
            hi = np.quantile(x, 0.975, axis=0)
            return lo, hi

        pos_lo, pos_hi = band(irf_pos_draws)
        neg_lo, neg_hi = band(irf_neg_draws)

        # Plot
        h = np.arange(HORIZON+1)
        fig, ax = plt.subplots(figsize=(10, 4))

        ax.plot(h, irf_pos_hat, label="IRF NASCDI⁺ (intensification)")
        ax.fill_between(h, pos_lo, pos_hi, alpha=0.25)

        ax.plot(h, irf_neg_hat, label="IRF NASCDI⁻ (easing)")
        ax.fill_between(h, neg_lo, neg_hi, alpha=0.25)

        ax.axhline(0, linestyle="--", linewidth=1)
        ax.set_xlabel("Weeks")
        ax.set_ylabel("Terminal price response (level)")
        ax.set_title(f"IRF with 95% bands — {market} | {variety} | G{grade}")
        ax.legend()
        fig.tight_layout()

        safe_name = f"{market}_{variety}_{grade}".replace(" ", "_")
        out = os.path.join(OUT_DIR, f"irf_bands_{safe_name}.png")
        fig.savefig(out, dpi=220)
        plt.close(fig)

        print("✅ Saved:", out)

if __name__ == "__main__":
    main()
