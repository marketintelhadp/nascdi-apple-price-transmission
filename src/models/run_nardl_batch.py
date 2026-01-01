import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan

# -----------------------
# Config
# -----------------------
IN_DIR = "data/model"
OUT_DIR = "results/nardl"
PLOT_DIR = os.path.join(OUT_DIR, "plots")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

MAX_LAGS_Y = 4        # lags for dependent variable (weekly)
MAX_LAGS_X = 4        # lags for regressors
MIN_OBS = 60          # minimum usable observations

# -----------------------
# Helpers
# -----------------------
def parse_name(fname: str):
    # weekly_model_<Market>_<Variety>_<Grade>_posneg.csv
    base = os.path.basename(fname).replace("_posneg.csv", "")
    m = re.match(r"weekly_model_(.+)_(American|Delicious)_(A|B)", base)
    if not m:
        return ("Unknown", "Unknown", "Unknown")
    return m.group(1), m.group(2), m.group(3)

def safe_adf(x):
    x = pd.Series(x).dropna()
    if len(x) < 20:
        return np.nan, np.nan
    stat, pval, *_ = adfuller(x, autolag="AIC")
    return stat, pval

def make_lags(df, col, max_lag):
    for L in range(1, max_lag + 1):
        df[f"{col}_L{L}"] = df[col].shift(L)
    return df

def build_ardl_design(df):
    """
    NARDL-style ARDL in levels + short-run differences via ECM representation:

    Δy_t = c + φ*(y_{t-1} - θ1*x_{t-1} - θ2*p_{t-1} - θ3*n_{t-1}) 
           + Σ α_i Δy_{t-i} + Σ β_i Δx_{t-i} + Σ γ_i Δp_{t-i} + Σ δ_i Δn_{t-i} + e_t

    Here:
      y  = avg_price_term
      x  = avg_price_prod
      p  = NASCDI_pos
      n  = NASCDI_neg
    """
    d = df.sort_values("week_end").copy()
    y = "avg_price_term"
    x = "avg_price_prod"
    p = "NASCDI_pos"
    n = "NASCDI_neg"

    # Differences
    d["dy"] = d[y].diff()
    d["dx"] = d[x].diff()
    d["dp"] = d[p].diff()
    d["dn"] = d[n].diff()

    # Lagged levels for ECM term
    d["y_L1"] = d[y].shift(1)
    d["x_L1"] = d[x].shift(1)
    d["p_L1"] = d[p].shift(1)
    d["n_L1"] = d[n].shift(1)

    # Lagged differences (short-run)
    for L in range(1, MAX_LAGS_Y + 1):
        d[f"dy_L{L}"] = d["dy"].shift(L)
    for L in range(0, MAX_LAGS_X + 1):
        d[f"dx_L{L}"] = d["dx"].shift(L)
        d[f"dp_L{L}"] = d["dp"].shift(L)
        d[f"dn_L{L}"] = d["dn"].shift(L)

    # Drop rows with missing due to differencing/lags
    cols_needed = ["dy", "y_L1", "x_L1", "p_L1", "n_L1"] + \
                  [f"dy_L{L}" for L in range(1, MAX_LAGS_Y + 1)] + \
                  [f"dx_L{L}" for L in range(0, MAX_LAGS_X + 1)] + \
                  [f"dp_L{L}" for L in range(0, MAX_LAGS_X + 1)] + \
                  [f"dn_L{L}" for L in range(0, MAX_LAGS_X + 1)]

    d2 = d.dropna(subset=cols_needed).copy()

    Y = d2["dy"]
    Xcols = ["y_L1", "x_L1", "p_L1", "n_L1"] + \
            [f"dy_L{L}" for L in range(1, MAX_LAGS_Y + 1)] + \
            [f"dx_L{L}" for L in range(0, MAX_LAGS_X + 1)] + \
            [f"dp_L{L}" for L in range(0, MAX_LAGS_X + 1)] + \
            [f"dn_L{L}" for L in range(0, MAX_LAGS_X + 1)]

    X = add_constant(d2[Xcols], has_constant="add")
    return d2, Y, X

def bounds_like_summary(model):
    """
    A pragmatic "bounds-like" indicator:
    - We focus on significance of (y_{t-1}, x_{t-1}, p_{t-1}, n_{t-1})
    - True Pesaran bounds requires critical values; we report:
      * joint F-test p-value for lagged levels block
      * ECT sign via y_L1 coefficient
    """
    # block F-test: y_L1 = x_L1 = p_L1 = n_L1 = 0
    R = np.zeros((4, model.params.shape[0]))
    param_names = model.params.index.tolist()
    for i, nm in enumerate(["y_L1", "x_L1", "p_L1", "n_L1"]):
        if nm in param_names:
            R[i, param_names.index(nm)] = 1.0
    ftest = model.f_test(R)
    return float(ftest.fvalue), float(ftest.pvalue)

def long_run_effects(model):
    """
    From ECM:
      dy = ... + a*y_L1 + b*x_L1 + c*p_L1 + d*n_L1 + ...
    Long-run coefficients (θ) are:
      θx = -b/a, θp = -c/a, θn = -d/a
    Need a = coefficient on y_L1 (should be negative for adjustment)
    """
    a = model.params.get("y_L1", np.nan)
    b = model.params.get("x_L1", np.nan)
    c = model.params.get("p_L1", np.nan)
    d = model.params.get("n_L1", np.nan)
    if np.isnan(a) or a == 0:
        return (np.nan, np.nan, np.nan)
    return (-b/a, -c/a, -d/a)

def plot_series(df, title, outpath):
    plt.figure(figsize=(10, 3))
    plt.plot(df["week_end"], df["avg_price_prod"], label="Producer")
    plt.plot(df["week_end"], df["avg_price_term"], label="Azadpur")
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def plot_residuals(resid, title, outpath):
    plt.figure(figsize=(10, 3))
    plt.plot(resid)
    plt.axhline(0, linestyle="--")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

# -----------------------
# Main batch run
# -----------------------
rows = []
diag_rows = []

files = sorted(glob.glob(os.path.join(IN_DIR, "weekly_model_*_posneg.csv")))
if not files:
    raise FileNotFoundError(f"No posneg model files found in {IN_DIR}")

for fp in files:
    market, variety, grade = parse_name(fp)
    df = pd.read_csv(fp, parse_dates=["week_end"])

    # basic checks
    df = df.dropna(subset=["avg_price_prod", "avg_price_term", "NASCDI", "NASCDI_pos", "NASCDI_neg"])
    if len(df) < MIN_OBS:
        print(f"SKIP {os.path.basename(fp)} (too short: {len(df)})")
        continue

    # Plot raw series
    plot_series(df, f"{market}-{variety}-{grade}", os.path.join(PLOT_DIR, f"series_{market}_{variety}_{grade}.png"))

    # Stationarity quick check
    adf_y = safe_adf(df["avg_price_term"])
    adf_x = safe_adf(df["avg_price_prod"])

    # Build ECM/NARDL design
    d2, Y, X = build_ardl_design(df)
    if len(d2) < MIN_OBS:
        print(f"SKIP {os.path.basename(fp)} after lags (too short: {len(d2)})")
        continue

    model = OLS(Y, X).fit()

    # Bounds-like
    fval, fpval = bounds_like_summary(model)

    # Long-run effects
    lr_x, lr_pos, lr_neg = long_run_effects(model)

    # Diagnostics
    resid = model.resid
    lb = acorr_ljungbox(resid, lags=[8], return_df=True)
    lb_p = float(lb["lb_pvalue"].iloc[0])

    bp = het_breuschpagan(resid, model.model.exog)
    bp_p = float(bp[1])

    # Residual plot
    plot_residuals(resid, f"Residuals {market}-{variety}-{grade}", os.path.join(PLOT_DIR, f"resid_{market}_{variety}_{grade}.png"))

    # Collect summary row
    rows.append({
        "market": market,
        "variety": variety,
        "grade": grade,
        "n_obs": len(d2),
        "ECT_y_L1": model.params.get("y_L1", np.nan),
        "ECT_pvalue": model.pvalues.get("y_L1", np.nan),
        "LR_pass_through_prod": lr_x,
        "LR_effect_NASCDI_pos": lr_pos,
        "LR_effect_NASCDI_neg": lr_neg,
        "bounds_F": fval,
        "bounds_F_pvalue": fpval,
        "ADF_term_pvalue": adf_y[1],
        "ADF_prod_pvalue": adf_x[1],
        "R2": model.rsquared
    })

    diag_rows.append({
        "market": market,
        "variety": variety,
        "grade": grade,
        "ljungbox_p(8)": lb_p,
        "bp_heterosk_p": bp_p,
        "AIC": model.aic,
        "BIC": model.bic
    })
if not rows:
    raise RuntimeError(
        "No series were successfully estimated. "
        "Check IN_DIR, file pattern, and whether posneg files contain rows."
    )
# Save outputs
res_df = pd.DataFrame(rows).sort_values(["market", "variety", "grade"])
diag_df = pd.DataFrame(diag_rows).sort_values(["market", "variety", "grade"])

res_df.to_csv(os.path.join(OUT_DIR, "nardl_results_table.csv"), index=False)
diag_df.to_csv(os.path.join(OUT_DIR, "nardl_diagnostics.csv"), index=False)

print("✅ NARDL batch run complete.")
print("Saved:", os.path.join(OUT_DIR, "nardl_results_table.csv"))
print("Saved:", os.path.join(OUT_DIR, "nardl_diagnostics.csv"))
print("Plots in:", PLOT_DIR)
