import os
import glob
import re
import numpy as np
import pandas as pd

from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

# ============================================================
# CONFIG
# ============================================================
IN_DIR = "data/model"
OUT_DIR = "results/nardl"
os.makedirs(OUT_DIR, exist_ok=True)

# Lags must match your estimation script
MAX_LAGS_Y = 4
MAX_LAGS_X = 4
MIN_OBS = 25

# HAC (Newey–West) bandwidth (weekly). 8 is a common choice.
# You may also set it to MAX_LAGS_Y + MAX_LAGS_X + 2.
HAC_MAXLAGS = 8


# ============================================================
# HELPERS
# ============================================================
def parse_name(fname: str):
    """
    weekly_model_<Market>_<Variety>_<Grade>_posneg.csv
    Works even if market contains underscores.
    """
    base = os.path.basename(fname).replace("_posneg.csv", "")
    m = re.match(r"weekly_model_(.+)_(American|Delicious)_(A|B)", base)
    if not m:
        return ("Unknown", "Unknown", "Unknown")
    return m.group(1), m.group(2), m.group(3)


def infer_yx_columns(df: pd.DataFrame):
    """
    Choose y and x columns robustly.
    Preference:
      y: log_price_term if exists else avg_price_term
      x: log_price_prod if exists else avg_price_prod
    """
    if "log_price_term" in df.columns and "log_price_prod" in df.columns:
        return "log_price_term", "log_price_prod"
    if "avg_price_term" in df.columns and "avg_price_prod" in df.columns:
        return "avg_price_term", "avg_price_prod"
    raise ValueError(
        f"Could not infer y/x columns. Found columns: {list(df.columns)}"
    )


def build_ecm_design(df: pd.DataFrame, y: str, x: str):
    """
    ECM/NARDL-style design (same structure as your run_nardl_batch.py):

    Δy_t = c + a*y_{t-1} + b*x_{t-1} + c1*p_{t-1} + c2*n_{t-1}
           + Σ α_i Δy_{t-i}
           + Σ β_i Δx_{t-i}
           + Σ γ_i Δp_{t-i}
           + Σ δ_i Δn_{t-i} + e_t

    where:
      p = NASCDI_pos
      n = NASCDI_neg
    """
    d = df.sort_values("week_end").copy()
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

    # Lagged differences
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
    return d2, Y, X


def long_run_effects(params):
    """
    From ECM:
      dy = ... + a*y_L1 + b*x_L1 + c*p_L1 + d*n_L1 + ...
    Long-run:
      LR_x   = -b/a
      LR_pos = -c/a
      LR_neg = -d/a
    """
    a = params.get("y_L1", np.nan)
    b = params.get("x_L1", np.nan)
    c = params.get("p_L1", np.nan)
    d = params.get("n_L1", np.nan)
    if np.isnan(a) or a == 0:
        return (np.nan, np.nan, np.nan)
    return (-b / a, -c / a, -d / a)


# ============================================================
# MAIN
# ============================================================
def main():
    files = sorted(glob.glob(os.path.join(IN_DIR, "weekly_model_*_posneg.csv")))
    if not files:
        raise FileNotFoundError(
            f"No files found in {IN_DIR} matching weekly_model_*_posneg.csv"
        )

    param_rows = []
    summary_rows = []

    for fp in files:
        market, variety, grade = parse_name(fp)
        df = pd.read_csv(fp, parse_dates=["week_end"])

        # Required NASCDI columns
        for c in ["NASCDI_pos", "NASCDI_neg"]:
            if c not in df.columns:
                print(f"SKIP {os.path.basename(fp)} (missing {c})")
                continue

        # Infer y/x
        try:
            y, x = infer_yx_columns(df)
        except ValueError as e:
            print(f"SKIP {os.path.basename(fp)} ({e})")
            continue

        # Drop key missing
        df = df.dropna(subset=[y, x, "NASCDI_pos", "NASCDI_neg"]).copy()

        # Build design
        d2, Y, X = build_ecm_design(df, y=y, x=x)
        if len(d2) < MIN_OBS:
            print(f"SKIP {os.path.basename(fp)} after lags (too short: {len(d2)})")
            continue

        # Fit OLS (coefficients are identical regardless of robust SE choice)
        model = OLS(Y, X).fit()

        # HAC(Newey–West) robust inference
        # NOTE: coefficients same; only SE/t/p change.
        model_hac = model.get_robustcov_results(cov_type="HAC", maxlags=HAC_MAXLAGS)

        # Long-run effects (point estimates)
        lr_x, lr_pos, lr_neg = long_run_effects(model.params)

        # Store key summary with BOTH p-values (OLS and HAC)
        summary_rows.append({
            "market": market,
            "variety": variety,
            "grade": grade,
            "y_col": y,
            "x_col": x,
            "n_obs": int(len(d2)),
            "ECT_y_L1_coef": float(model.params.get("y_L1", np.nan)),
            "ECT_p_OLS": float(model.pvalues.get("y_L1", np.nan)),
            "ECT_p_HAC": float(model_hac.pvalues[model_hac.model.exog_names.index("y_L1")]) if "y_L1" in model_hac.model.exog_names else np.nan,
            "x_L1_p_OLS": float(model.pvalues.get("x_L1", np.nan)),
            "p_L1_p_OLS": float(model.pvalues.get("p_L1", np.nan)),
            "n_L1_p_OLS": float(model.pvalues.get("n_L1", np.nan)),
            "x_L1_p_HAC": float(model_hac.pvalues[model_hac.model.exog_names.index("x_L1")]) if "x_L1" in model_hac.model.exog_names else np.nan,
            "p_L1_p_HAC": float(model_hac.pvalues[model_hac.model.exog_names.index("p_L1")]) if "p_L1" in model_hac.model.exog_names else np.nan,
            "n_L1_p_HAC": float(model_hac.pvalues[model_hac.model.exog_names.index("n_L1")]) if "n_L1" in model_hac.model.exog_names else np.nan,
            "LR_pass_through_prod": float(lr_x),
            "LR_effect_NASCDI_pos": float(lr_pos),
            "LR_effect_NASCDI_neg": float(lr_neg),
            "R2": float(model.rsquared),
            "AIC": float(model.aic),
            "BIC": float(model.bic),
            "file": os.path.basename(fp),
        })

        # Parameter-level table (all params with OLS + HAC SE/p)
        for name in model.params.index:
            # OLS
            coef = float(model.params[name])
            se_ols = float(model.bse[name])
            p_ols = float(model.pvalues[name])

            # HAC
            # robust results use positional indexing via exog_names
            if name in model_hac.model.exog_names:
                j = model_hac.model.exog_names.index(name)
                se_hac = float(model_hac.bse[j])
                p_hac = float(model_hac.pvalues[j])
                t_hac = float(model_hac.tvalues[j])
            else:
                se_hac = np.nan
                p_hac = np.nan
                t_hac = np.nan

            param_rows.append({
                "market": market,
                "variety": variety,
                "grade": grade,
                "param": name,
                "coef": coef,
                "se_OLS": se_ols,
                "p_OLS": p_ols,
                "t_HAC": t_hac,
                "se_HAC": se_hac,
                "p_HAC": p_hac,
                "n_obs": int(len(d2)),
                "file": os.path.basename(fp),
            })

    if not summary_rows:
        raise RuntimeError("No models were successfully processed for HAC inference.")

    # Save outputs
    summary_df = pd.DataFrame(summary_rows).sort_values(["market", "variety", "grade"])
    params_df = pd.DataFrame(param_rows).sort_values(["market", "variety", "grade", "param"])

    out_summary = os.path.join(OUT_DIR, "nardl_hac_summary.csv")
    out_params = os.path.join(OUT_DIR, "nardl_hac_params.csv")

    summary_df.to_csv(out_summary, index=False)
    params_df.to_csv(out_params, index=False)

    print("✅ HAC(Newey–West) inference complete.")
    print("Saved:", out_summary)
    print("Saved:", out_params)
    print(f"HAC maxlags used: {HAC_MAXLAGS}")


if __name__ == "__main__":
    main()
