import os
import glob
import re
import numpy as np
import pandas as pd

from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan

# -----------------------
# Config
# -----------------------
IN_DIR = "data/model"         # <-- IMPORTANT: use rebuilt folder
OUT_DIR = "results/nardl"
PLOT_DIR = os.path.join(OUT_DIR, "plots")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

MAX_LAGS_Y = 4
MAX_LAGS_X = 4
MIN_OBS_AFTER_LAGS = 60  # stricter so estimation is stable

def parse_name(fname: str):
    """
    weekly_model_<producer>_<variety>_<grade>_posneg.csv
    producer can contain underscores.
    """
    base = os.path.basename(fname).replace("_posneg.csv", "")
    m = re.match(r"weekly_model_(.+)_(American|Delicious|Maharaji)_(A|B)$", base)
    if not m:
        return ("Unknown", "Unknown", "Unknown")
    return m.group(1), m.group(2), m.group(3)

def build_ardl_design(df):
    """
    ECM/NARDL-style design:
    dy_t = c + a*y_{t-1} + b*x_{t-1} + c1*p_{t-1} + c2*n_{t-1}
           + sum lagged dy + sum lagged dx + sum lagged dp + sum lagged dn + e
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

    # Lagged levels
    d["y_L1"] = d[y].shift(1)
    d["x_L1"] = d[x].shift(1)
    d["p_L1"] = d[p].shift(1)
    d["n_L1"] = d[n].shift(1)

    # Lagged diffs
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

def bounds_like_summary(model):
    # joint test of lagged levels block
    names = model.params.index.tolist()
    R = np.zeros((4, len(names)))
    for i, nm in enumerate(["y_L1", "x_L1", "p_L1", "n_L1"]):
        if nm in names:
            R[i, names.index(nm)] = 1.0
    ftest = model.f_test(R)
    return float(ftest.fvalue), float(ftest.pvalue)

def long_run_effects(model):
    a = model.params.get("y_L1", np.nan)
    b = model.params.get("x_L1", np.nan)
    c = model.params.get("p_L1", np.nan)
    d = model.params.get("n_L1", np.nan)
    if np.isnan(a) or a == 0:
        return (np.nan, np.nan, np.nan)
    return (-b/a, -c/a, -d/a)

def main():
    files = sorted(glob.glob(os.path.join(IN_DIR, "weekly_model_*_posneg.csv")))
    if not files:
        raise FileNotFoundError(f"No model files found in {IN_DIR}")

    rows = []
    diag = []
    skipped = 0

    for fp in files:
        market, variety, grade = parse_name(fp)
        df = pd.read_csv(fp, parse_dates=["week_end"])

        required = ["avg_price_prod", "avg_price_term", "NASCDI", "NASCDI_pos", "NASCDI_neg"]
        miss = [c for c in required if c not in df.columns]
        if miss:
            print(f"SKIP {os.path.basename(fp)} (missing {miss})")
            skipped += 1
            continue

        # Drop NA
        df = df.dropna(subset=required).sort_values("week_end")
        if len(df) < MIN_OBS_AFTER_LAGS:
            print(f"SKIP {os.path.basename(fp)} (too short pre-lags: {len(df)})")
            skipped += 1
            continue

        # Build design (will drop additional rows)
        d2, Y, X = build_ardl_design(df)
        if len(d2) < MIN_OBS_AFTER_LAGS:
            print(f"SKIP {os.path.basename(fp)} (too short after lags: {len(d2)})")
            skipped += 1
            continue

        model = OLS(Y, X).fit()

        fval, fpval = bounds_like_summary(model)
        lr_x, lr_pos, lr_neg = long_run_effects(model)

        resid = model.resid
        lb = acorr_ljungbox(resid, lags=[8], return_df=True)
        lb_p = float(lb["lb_pvalue"].iloc[0])

        bp = het_breuschpagan(resid, model.model.exog)
        bp_p = float(bp[1])

        rows.append({
            "producer_market": market,
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
            "R2": model.rsquared,
            "AIC": model.aic,
            "BIC": model.bic,
        })

        diag.append({
            "producer_market": market,
            "variety": variety,
            "grade": grade,
            "ljungbox_p(8)": lb_p,
            "bp_heterosk_p": bp_p
        })

    if not rows:
        raise RuntimeError(
            "No series estimated. Most likely your merge produced NASCDI NaNs due to week_end mismatch.\n"
            "Rebuild using src/data/build_weekly_prices_and_posneg.py and ensure IN_DIR points to data/model_posneg."
        )

    res_df = pd.DataFrame(rows).sort_values(["producer_market", "variety", "grade"])
    diag_df = pd.DataFrame(diag).sort_values(["producer_market", "variety", "grade"])

    res_out = os.path.join(OUT_DIR, "nardl_results_table.csv")
    diag_out = os.path.join(OUT_DIR, "nardl_diagnostics.csv")

    res_df.to_csv(res_out, index=False)
    diag_df.to_csv(diag_out, index=False)

    print(f"✅ Estimated: {len(res_df)} | Skipped: {skipped}")
    print(f"Saved: {res_out}")
    print(f"Saved: {diag_out}")

if __name__ == "__main__":
    main()
