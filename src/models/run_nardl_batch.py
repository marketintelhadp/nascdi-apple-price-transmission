# src/models/run_nardl_batch.py
# ---------------------------------------------------------
# Corrected + hardened NARDL batch runner (event-based NASCDI)
# Fixes included:
#  1) Robust NASCDI merge if missing in model files
#  2) Robust parsing of file name (market/variety/grade)
#  3) Handles Maharaji too (safe for expansion)
#  4) Guards against constant / all-zero regressors (dp/dn)
#  5) Avoids “no series estimated” by printing WHY each file is skipped
#  6) Ensures the POS/NEG series are usable (levels should vary)
# ---------------------------------------------------------

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
MIN_OBS = 25          # minimum usable observations AFTER lags

NASCDI_WEEKLY_PATH = "data/nascdi/nascdi_event_weekly.csv"
_nascdi_weekly = None


# -----------------------
# Utilities
# -----------------------
def load_weekly_nascdi() -> pd.DataFrame:
    """
    Load weekly NASCDI master series, normalize the date column to 'week_end',
    and return ['week_end','NASCDI'] unique by week_end.
    """
    global _nascdi_weekly
    if _nascdi_weekly is not None:
        return _nascdi_weekly

    t = pd.read_csv(NASCDI_WEEKLY_PATH)

    if "week_end" in t.columns:
        t["week_end"] = pd.to_datetime(t["week_end"])
    elif "date" in t.columns:
        t["week_end"] = pd.to_datetime(t["date"])
    else:
        raise ValueError("NASCDI weekly file must have 'date' or 'week_end' column")

    if "NASCDI" not in t.columns:
        raise ValueError("NASCDI weekly file must contain 'NASCDI' column")

    _nascdi_weekly = (
        t[["week_end", "NASCDI"]]
        .dropna(subset=["week_end", "NASCDI"])
        .drop_duplicates(subset=["week_end"])
        .sort_values("week_end")
        .reset_index(drop=True)
    )
    return _nascdi_weekly


def parse_name(fname: str):
    """
    Expected: weekly_model_<Market>_<Variety>_<Grade>_posneg.csv
    Market can contain underscores.
    """
    base = os.path.basename(fname)
    base = base.replace("weekly_model_", "").replace("_posneg.csv", "")

    # Split from the right to safely keep underscores in market names
    parts = base.rsplit("_", 2)
    if len(parts) != 3:
        return ("Unknown", "Unknown", "Unknown")

    market, variety, grade = parts[0], parts[1], parts[2]
    # allow future expansion
    if variety not in {"American", "Delicious", "Maharaji"}:
        variety = "Unknown"
    if grade not in {"A", "B"}:
        grade = "Unknown"
    return market, variety, grade


def safe_adf(x):
    x = pd.Series(x).dropna()
    if len(x) < 20:
        return np.nan, np.nan
    stat, pval, *_ = adfuller(x, autolag="AIC")
    return stat, pval


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


def build_ardl_design(df: pd.DataFrame):
    """
    ECM-style NARDL design with:
      y  = avg_price_term
      x  = avg_price_prod
      p  = NASCDI_pos (level partial sum)
      n  = NASCDI_neg (level partial sum)

    Δy_t = c + a*y_{t-1} + b*x_{t-1} + c1*p_{t-1} + c2*n_{t-1}
           + Σ α_i Δy_{t-i} + Σ β_i Δx_{t-i} + Σ γ_i Δp_{t-i} + Σ δ_i Δn_{t-i} + e_t
    """
    d = df.sort_values("week_end").copy()

    y, x, p, n = "avg_price_term", "avg_price_prod", "NASCDI_pos", "NASCDI_neg"

    # Differences
    d["dy"] = d[y].diff()
    d["dx"] = d[x].diff()
    d["dp"] = d[p].diff()
    d["dn"] = d[n].diff()

    # Lagged levels
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

    cols_needed = (
        ["dy", "y_L1", "x_L1", "p_L1", "n_L1"]
        + [f"dy_L{L}" for L in range(1, MAX_LAGS_Y + 1)]
        + [f"dx_L{L}" for L in range(0, MAX_LAGS_X + 1)]
        + [f"dp_L{L}" for L in range(0, MAX_LAGS_X + 1)]
        + [f"dn_L{L}" for L in range(0, MAX_LAGS_X + 1)]
    )

    d2 = d.dropna(subset=cols_needed).copy()

    Y = d2["dy"]

    Xcols = (
        ["y_L1", "x_L1", "p_L1", "n_L1"]
        + [f"dy_L{L}" for L in range(1, MAX_LAGS_Y + 1)]
        + [f"dx_L{L}" for L in range(0, MAX_LAGS_X + 1)]
        + [f"dp_L{L}" for L in range(0, MAX_LAGS_X + 1)]
        + [f"dn_L{L}" for L in range(0, MAX_LAGS_X + 1)]
    )

    X = add_constant(d2[Xcols], has_constant="add")

    # Drop any columns that are all-NaN or constant (can break OLS / diagnostics)
    nunique = X.nunique(dropna=True)
    const_cols = [c for c in X.columns if c != "const" and nunique.get(c, 0) <= 1]
    if const_cols:
        X = X.drop(columns=const_cols)

    return d2, Y, X


def bounds_like_summary(model):
    """
    Block F-test on lagged levels: y_L1, x_L1, p_L1, n_L1
    """
    param_names = list(model.params.index)

    level_vars = ["y_L1", "x_L1", "p_L1", "n_L1"]
    present = [v for v in level_vars if v in param_names]
    if len(present) < 2:
        return np.nan, np.nan

    R = np.zeros((len(present), len(param_names)))
    for i, nm in enumerate(present):
        R[i, param_names.index(nm)] = 1.0

    ftest = model.f_test(R)
    return float(ftest.fvalue), float(ftest.pvalue)


def long_run_effects(model):
    """
    LR: θx=-b/a, θp=-c/a, θn=-d/a from coefficients on x_L1,p_L1,n_L1 over y_L1
    """
    a = model.params.get("y_L1", np.nan)
    b = model.params.get("x_L1", np.nan)
    c = model.params.get("p_L1", np.nan)
    d = model.params.get("n_L1", np.nan)

    if np.isnan(a) or a == 0:
        return (np.nan, np.nan, np.nan)
    return (-b / a, -c / a, -d / a)


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

    # --- Required columns check ---
    required = ["week_end", "avg_price_prod", "avg_price_term", "NASCDI_pos", "NASCDI_neg"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"SKIP {os.path.basename(fp)} (missing columns: {missing})")
        continue

    # --- NASCDI merge if missing ---
    if "NASCDI" not in df.columns:
        nascdi_w = load_weekly_nascdi()
        before = len(df)
        df = df.merge(nascdi_w, on="week_end", how="left")
        after = len(df)
        if after != before:
            print(f"NOTE: row count changed after NASCDI merge for {os.path.basename(fp)}: {before} -> {after}")

    # --- Drop NA in core vars (zeros allowed; NA not allowed) ---
    df = df.dropna(subset=["avg_price_prod", "avg_price_term", "NASCDI", "NASCDI_pos", "NASCDI_neg"])

    # --- Quick variability guards (prevents “all zeros” issues) ---
    if df["avg_price_term"].nunique() < 5 or df["avg_price_prod"].nunique() < 5:
        print(f"SKIP {os.path.basename(fp)} (insufficient price variation)")
        continue

    # POS/NEG are cumulative partial sums → should vary at least sometimes
    if df["NASCDI_pos"].nunique() <= 1 and df["NASCDI_neg"].nunique() <= 1:
        print(f"SKIP {os.path.basename(fp)} (NASCDI_pos and NASCDI_neg are constant; rebuild POS/NEG globally)")
        continue

    if len(df) < MIN_OBS:
        print(f"SKIP {os.path.basename(fp)} (too short pre-lags: {len(df)})")
        continue

    # Plots
    plot_series(df, f"{market}-{variety}-{grade}",
                os.path.join(PLOT_DIR, f"series_{market}_{variety}_{grade}.png"))

    # Stationarity quick check
    adf_y = safe_adf(df["avg_price_term"])
    adf_x = safe_adf(df["avg_price_prod"])

    # Build design
    d2, Y, X = build_ardl_design(df)

    if len(d2) < MIN_OBS:
        print(f"SKIP {os.path.basename(fp)} after lags (too short: {len(d2)})")
        continue

    # OLS
    model = OLS(Y, X).fit()

    # Bounds-like
    fval, fpval = bounds_like_summary(model)

    # Long-run effects
    lr_x, lr_pos, lr_neg = long_run_effects(model)

    # Diagnostics
    resid = model.resid

    try:
        lb = acorr_ljungbox(resid, lags=[8], return_df=True)
        lb_p = float(lb["lb_pvalue"].iloc[0])
    except Exception:
        lb_p = np.nan

    try:
        bp = het_breuschpagan(resid, model.model.exog)
        bp_p = float(bp[1])
    except Exception:
        bp_p = np.nan

    plot_residuals(resid, f"Residuals {market}-{variety}-{grade}",
                   os.path.join(PLOT_DIR, f"resid_{market}_{variety}_{grade}.png"))

    # Save row
    rows.append({
        "market": market,
        "variety": variety,
        "grade": grade,
        "n_obs": int(len(d2)),
        "ECT_y_L1": float(model.params.get("y_L1", np.nan)),
        "ECT_pvalue": float(model.pvalues.get("y_L1", np.nan)),
        "LR_pass_through_prod": float(lr_x) if lr_x == lr_x else np.nan,
        "LR_effect_NASCDI_pos": float(lr_pos) if lr_pos == lr_pos else np.nan,
        "LR_effect_NASCDI_neg": float(lr_neg) if lr_neg == lr_neg else np.nan,
        "bounds_F": float(fval) if fval == fval else np.nan,
        "bounds_F_pvalue": float(fpval) if fpval == fpval else np.nan,
        "ADF_term_pvalue": float(adf_y[1]) if adf_y[1] == adf_y[1] else np.nan,
        "ADF_prod_pvalue": float(adf_x[1]) if adf_x[1] == adf_x[1] else np.nan,
        "R2": float(model.rsquared)
    })

    diag_rows.append({
        "market": market,
        "variety": variety,
        "grade": grade,
        "ljungbox_p(8)": lb_p,
        "bp_heterosk_p": bp_p,
        "AIC": float(model.aic),
        "BIC": float(model.bic)
    })

if not rows:
    raise RuntimeError(
        "No series were successfully estimated.\n"
        "Most common causes:\n"
        "  1) Individual posneg files are empty / too short after lags\n"
        "  2) NASCDI_pos and NASCDI_neg are constant (POS/NEG built group-wise wrongly)\n"
        "  3) Missing NASCDI in files and weekly NASCDI could not be merged\n"
        "Fix: rebuild POS/NEG globally in the NASCDI weekly series, then merge into all model files."
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
