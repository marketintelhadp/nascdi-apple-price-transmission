# ============================================================
# Residual diagnostics for NARDL/ECM batch (cointegration support)
# Works with weekly_model_*_posneg.csv files (prices + NASCDI_pos/neg)
# ============================================================

import os
import glob
import re
import numpy as np
import pandas as pd

from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import (
    acorr_ljungbox,
    het_breuschpagan,
    het_white,
    acorr_breusch_godfrey,
    normal_ad,
    het_arch,
    linear_reset
)
from statsmodels.stats.stattools import jarque_bera

# -----------------------
# Config
# -----------------------
DATA_MODEL_DIR = os.path.join("data", "model")
OUT_DIR = os.path.join("results", "nardl")
OUT_PATH = os.path.join(OUT_DIR, "residual_diagnostics_cointegration.csv")

MAX_LAGS_Y = 4
MAX_LAGS_X = 4
MIN_OBS = 25

os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------
# Helpers
# -----------------------
def parse_name(fname: str):
    base = os.path.basename(fname).replace("_posneg.csv", "")
    m = re.match(r"weekly_model_(.+)_(American|Delicious)_(A|B)", base)
    if not m:
        return ("Unknown", "Unknown", "Unknown")
    return m.group(1), m.group(2), m.group(3)

def build_ardl_design(df):
    """
    Same ECM design you used in run_nardl_batch.py

    Δy_t = c + a*y_{t-1} + b*x_{t-1} + c1*p_{t-1} + c2*n_{t-1}
           + Σ α_i Δy_{t-i} + Σ β_i Δx_{t-i}
           + Σ γ_i Δp_{t-i} + Σ δ_i Δn_{t-i} + e_t

    where:
      y = avg_price_term
      x = avg_price_prod
      p = NASCDI_pos
      n = NASCDI_neg
    """
    d = df.sort_values("week_end").copy()

    y, x, p, n = "avg_price_term", "avg_price_prod", "NASCDI_pos", "NASCDI_neg"

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

def safe_adf_resid(resid):
    resid = pd.Series(resid).dropna()
    if len(resid) < 20:
        return (np.nan, np.nan)
    stat, pval, *_ = adfuller(resid, autolag="AIC")
    return (float(stat), float(pval))

# -----------------------
# Main
# -----------------------
def main():
    files = sorted(glob.glob(os.path.join(DATA_MODEL_DIR, "weekly_model_*_posneg.csv")))
    if not files:
        raise FileNotFoundError(
            f"No files found in {DATA_MODEL_DIR} matching weekly_model_*_posneg.csv"
        )

    rows = []

    for fp in files:
        market, variety, grade = parse_name(fp)
        df = pd.read_csv(fp, parse_dates=["week_end"])

        required = ["avg_price_prod", "avg_price_term", "NASCDI_pos", "NASCDI_neg"]
        if any(c not in df.columns for c in required):
            print(f"SKIP {os.path.basename(fp)} (missing required cols)")
            continue

        d2, Y, X = build_ardl_design(df)
        if len(d2) < MIN_OBS:
            print(f"SKIP {os.path.basename(fp)} after lags (n={len(d2)})")
            continue

        model = OLS(Y, X).fit()
        resid = model.resid

        # --- Serial correlation ---
        # Breusch-Godfrey (up to lag 8; weekly data)
        try:
            bg = acorr_breusch_godfrey(model, nlags=8)
            bg_lm_stat, bg_lm_p = float(bg[0]), float(bg[1])
        except Exception:
            bg_lm_stat, bg_lm_p = np.nan, np.nan

        # Ljung-Box Q test
        try:
            lb = acorr_ljungbox(resid, lags=[8], return_df=True)
            lb_stat = float(lb["lb_stat"].iloc[0])
            lb_p = float(lb["lb_pvalue"].iloc[0])
        except Exception:
            lb_stat, lb_p = np.nan, np.nan

        # --- Heteroskedasticity ---
        try:
            bp = het_breuschpagan(resid, model.model.exog)
            bp_stat, bp_p = float(bp[0]), float(bp[1])
        except Exception:
            bp_stat, bp_p = np.nan, np.nan

        try:
            wt = het_white(resid, model.model.exog)
            wt_stat, wt_p = float(wt[0]), float(wt[1])
        except Exception:
            wt_stat, wt_p = np.nan, np.nan

        # --- ARCH effects ---
        try:
            arch = het_arch(resid, nlags=8)
            arch_stat, arch_p = float(arch[0]), float(arch[1])
        except Exception:
            arch_stat, arch_p = np.nan, np.nan

        # --- Normality ---
        try:
            jb_stat, jb_p, _, _ = jarque_bera(resid)
            jb_stat, jb_p = float(jb_stat), float(jb_p)
        except Exception:
            jb_stat, jb_p = np.nan, np.nan

        # --- Functional form (RESET) ---
        try:
            reset_res = linear_reset(model, power=2, use_f=True)
            reset_f = float(reset_res.fvalue)
            reset_p = float(reset_res.pvalue)
        except Exception:
            reset_f, reset_p = np.nan, np.nan

        # --- Residual stationarity (cointegration support) ---
        adf_stat, adf_p = safe_adf_resid(resid)

        # ECT sign and significance
        ect = model.params.get("y_L1", np.nan)
        ect_p = model.pvalues.get("y_L1", np.nan)

        rows.append({
            "market": market,
            "variety": variety,
            "grade": grade,
            "n_obs_ecm": int(len(d2)),
            "ECT_y_L1": float(ect) if pd.notna(ect) else np.nan,
            "ECT_pvalue": float(ect_p) if pd.notna(ect_p) else np.nan,

            "BG_LM_stat(lag8)": bg_lm_stat,
            "BG_LM_pvalue(lag8)": bg_lm_p,

            "LjungBox_Q_stat(lag8)": lb_stat,
            "LjungBox_pvalue(lag8)": lb_p,

            "BP_stat": bp_stat,
            "BP_pvalue": bp_p,

            "White_stat": wt_stat,
            "White_pvalue": wt_p,

            "ARCH_LM_stat(lag8)": arch_stat,
            "ARCH_LM_pvalue(lag8)": arch_p,

            "JB_stat": jb_stat,
            "JB_pvalue": jb_p,

            "RESET_F": reset_f,
            "RESET_pvalue": reset_p,

            "ADF_resid_stat": adf_stat,
            "ADF_resid_pvalue": adf_p,

            "AIC": float(model.aic),
            "BIC": float(model.bic),
            "R2": float(model.rsquared),
        })

    out = pd.DataFrame(rows).sort_values(["market", "variety", "grade"])
    out.to_csv(OUT_PATH, index=False)
    print(f"✅ Saved residual diagnostics: {OUT_PATH}")
    print(f"Rows: {len(out)}")

if __name__ == "__main__":
    main()