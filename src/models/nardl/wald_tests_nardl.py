# ============================================================
# Wald Tests for NARDL Asymmetry (NASCDI+ vs NASCDI-)
# - Short-run asymmetry: equality of summed ΔNASCDI_pos vs ΔNASCDI_neg effects
# - Long-run asymmetry: equality of level NASCDI_pos vs NASCDI_neg coefficients
#
# Works with your weekly_model_*_posneg.csv files containing:
#   week_end, log_price_term, log_price_prod, NASCDI_pos, NASCDI_neg, ...
#
# Outputs:
#   results/nardl/tables/wald_asymmetry_tests.csv
# ============================================================

from __future__ import annotations

import re
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm


# -----------------------------
# Repo root finder (robust)
# -----------------------------
def find_repo_root(start: Path) -> Path:
    """
    Walk upwards until we find a folder that looks like the repo root.
    We accept a root that contains 'src' and 'data' (your structure).
    """
    cur = start.resolve()
    for _ in range(10):
        if (cur / "src").exists() and (cur / "data").exists():
            return cur
        cur = cur.parent
    # fallback to script's 4-level-up guess
    return start.resolve().parents[3]


# -----------------------------
# Statsmodels scalar safety
# -----------------------------
def _as_scalar(x) -> float:
    """Convert scalar / 1x1 / matrix-like into a python float safely."""
    arr = np.asarray(x).squeeze()
    return float(arr.ravel()[0])


# -----------------------------
# Helpers: infer columns from your files
# -----------------------------
def infer_columns(df: pd.DataFrame):
    """
    Your files show:
      log_price_term, log_price_prod, NASCDI_pos, NASCDI_neg
    We lock to these if present; otherwise do a safe fallback.
    """
    # Dependent variable: terminal log price
    if "log_price_term" in df.columns:
        y_col = "log_price_term"
    elif "price_term" in df.columns:
        y_col = "price_term"
    else:
        raise ValueError(f"Cannot find dependent price column. Found: {list(df.columns)}")

    # Producer price control (recommended)
    x_prod = None
    for c in ["log_price_prod", "producer_price", "price_prod", "log_price_producer"]:
        if c in df.columns:
            x_prod = c
            break

    # NASCDI pos/neg
    if "NASCDI_pos" in df.columns:
        x_pos = "NASCDI_pos"
    elif "nascdi_pos" in df.columns:
        x_pos = "nascdi_pos"
    else:
        raise ValueError(f"Cannot find NASCDI_pos column. Found: {list(df.columns)}")

    if "NASCDI_neg" in df.columns:
        x_neg = "NASCDI_neg"
    elif "nascdi_neg" in df.columns:
        x_neg = "nascdi_neg"
    else:
        raise ValueError(f"Cannot find NASCDI_neg column. Found: {list(df.columns)}")

    return y_col, x_prod, x_pos, x_neg


# -----------------------------
# Build NARDL-UECM design matrix
# -----------------------------
def make_uecm_design(
    df: pd.DataFrame,
    y: str,
    x_prod: str | None,
    x_pos: str,
    x_neg: str,
    p: int = 2,          # Δy lags
    q_prod: int = 2,     # Δx_prod lags
    q_pos: int = 2,      # Δx_pos lags
    q_neg: int = 2,      # Δx_neg lags
):
    """
    UECM form:
      Δy_t = c + φ*y_{t-1} + βp*x_prod_{t-1} + β+*x_pos_{t-1} + β-*x_neg_{t-1}
             + Σ α_i Δy_{t-i} + Σ γp_j Δx_prod_{t-j} + Σ γ+_j Δx_pos_{t-j} + Σ γ-_j Δx_neg_{t-j} + e_t

    Long-run asymmetry test: H0: β+ = β-
    Short-run asymmetry test: H0: Σ γ+_j = Σ γ-_j
    """

    d = df.copy()

    # Ensure numeric
    for c in [y, x_pos, x_neg] + ([x_prod] if x_prod else []):
        if c is None:
            continue
        d[c] = pd.to_numeric(d[c], errors="coerce")

    # Differences
    d["dy"] = d[y].diff()
    if x_prod:
        d["dx_prod"] = d[x_prod].diff()
    d["dx_pos"] = d[x_pos].diff()
    d["dx_neg"] = d[x_neg].diff()

    # Levels (t-1)
    d["y_l1"] = d[y].shift(1)
    if x_prod:
        d["xprod_l1"] = d[x_prod].shift(1)
    d["xpos_l1"] = d[x_pos].shift(1)
    d["xneg_l1"] = d[x_neg].shift(1)

    # Lagged Δy
    for i in range(1, p + 1):
        d[f"dy_l{i}"] = d["dy"].shift(i)

    # Lagged Δx (0..q) is standard in ARDL; for UECM we include Δx_t and lags
    # We'll include j=0..q for each regressor
    if x_prod:
        for j in range(0, q_prod + 1):
            d[f"dxprod_l{j}"] = d["dx_prod"].shift(j)
    for j in range(0, q_pos + 1):
        d[f"dxpos_l{j}"] = d["dx_pos"].shift(j)
    for j in range(0, q_neg + 1):
        d[f"dxneg_l{j}"] = d["dx_neg"].shift(j)

    # Assemble X columns
    X_cols = ["y_l1", "xpos_l1", "xneg_l1"]
    if x_prod:
        X_cols.insert(1, "xprod_l1")  # put producer control next to y_l1

    for i in range(1, p + 1):
        X_cols.append(f"dy_l{i}")

    if x_prod:
        for j in range(0, q_prod + 1):
            X_cols.append(f"dxprod_l{j}")

    for j in range(0, q_pos + 1):
        X_cols.append(f"dxpos_l{j}")
    for j in range(0, q_neg + 1):
        X_cols.append(f"dxneg_l{j}")

    # Drop NA from lagging
    use = d[["dy"] + X_cols].dropna().copy()

    Y = use["dy"]
    X = sm.add_constant(use[X_cols], has_constant="add")

    return Y, X, X_cols


# -----------------------------
# Wald tests
# -----------------------------
def wald_long_run(model, x_pos_level: str = "xpos_l1", x_neg_level: str = "xneg_l1"):
    """
    Long-run asymmetry in UECM:
      H0: β_pos(level) = β_neg(level)
    """
    params = model.params.index.tolist()
    if x_pos_level not in params or x_neg_level not in params:
        return np.nan, np.nan, "missing_level_terms"

    R = np.zeros((1, len(params)))
    R[0, params.index(x_pos_level)] = 1.0
    R[0, params.index(x_neg_level)] = -1.0
    q = np.array([0.0])

    try:
        test = model.wald_test((R, q), scalar=True)
    except TypeError:
        test = model.wald_test((R, q))

    return _as_scalar(test.statistic), _as_scalar(test.pvalue), "ok"


def wald_short_run(model, q_pos: int, q_neg: int):
    """
    Short-run asymmetry:
      H0: sum_{j=0..q_pos} γ_pos,j = sum_{j=0..q_neg} γ_neg,j

    Uses all ΔNASCDI terms included in the model:
      dxpos_l0..dxpos_lq_pos and dxneg_l0..dxneg_lq_neg
    """
    params = model.params.index.tolist()

    pos_terms = [f"dxpos_l{j}" for j in range(0, q_pos + 1) if f"dxpos_l{j}" in params]
    neg_terms = [f"dxneg_l{j}" for j in range(0, q_neg + 1) if f"dxneg_l{j}" in params]

    if len(pos_terms) == 0 or len(neg_terms) == 0:
        return np.nan, np.nan, "missing_diff_terms"

    R = np.zeros((1, len(params)))
    for t in pos_terms:
        R[0, params.index(t)] += 1.0
    for t in neg_terms:
        R[0, params.index(t)] += -1.0

    q = np.array([0.0])

    try:
        test = model.wald_test((R, q), scalar=True)
    except TypeError:
        test = model.wald_test((R, q))

    return _as_scalar(test.statistic), _as_scalar(test.pvalue), "ok"


# -----------------------------
# Parse market/variety/grade from filename
# -----------------------------
def parse_triplet_from_filename(path: Path):
    """
    Expected pattern in your repo (examples):
      weekly_model_Shopian_American_A_posneg.csv
      weekly_model_Sopore_Delicious_B_posneg.csv
    """
    name = path.stem
    # remove prefix/suffix
    name = re.sub(r"^weekly_model_", "", name)
    name = re.sub(r"_posneg$", "", name)
    parts = name.split("_")
    if len(parts) >= 3:
        market = parts[0]
        variety = parts[1]
        grade = parts[2]
    else:
        market, variety, grade = "NA", "NA", "NA"
    return market, variety, grade


# -----------------------------
# Main
# -----------------------------
def main():
    here = Path(__file__).resolve()
    ROOT = find_repo_root(here)

    DATA_MODEL_DIR = ROOT / "data" / "model"
    OUT_DIR = ROOT / "results" / "nardl" / "tables"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pattern = "weekly_model_*_posneg.csv"
    files = sorted(DATA_MODEL_DIR.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No files found at {DATA_MODEL_DIR} matching {pattern}\n"
            f"Check that your weekly model files are inside: {DATA_MODEL_DIR}"
        )

    rows = []

    # Fixed lags (safe + consistent). You can tune later.
    P = 2
    Q_PROD = 2
    Q_POS = 2
    Q_NEG = 2

    for f in files:
        df = pd.read_csv(f)

        # time column
        if "week_end" in df.columns:
            df["week_end"] = pd.to_datetime(df["week_end"], errors="coerce")
            df = df.sort_values("week_end").reset_index(drop=True)

        y_col, x_prod, x_pos, x_neg = infer_columns(df)
        market, variety, grade = parse_triplet_from_filename(f)

        # Build & fit
        Y, X, X_cols = make_uecm_design(
            df=df,
            y=y_col,
            x_prod=x_prod,
            x_pos=x_pos,
            x_neg=x_neg,
            p=P,
            q_prod=Q_PROD if x_prod else 0,
            q_pos=Q_POS,
            q_neg=Q_NEG,
        )

        # OLS with robust (HAC) SEs for weekly data (reviewer-friendly)
        # maxlags=4 is a standard conservative choice for weekly
        model = sm.OLS(Y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 4})

        # Wald tests
        w_lr, p_lr, lr_flag = wald_long_run(model)
        w_sr, p_sr, sr_flag = wald_short_run(model, q_pos=Q_POS, q_neg=Q_NEG)

        rows.append(
            {
                "file": f.name,
                "market": market,
                "variety": variety,
                "grade": grade,
                "y_col": y_col,
                "x_prod_col": x_prod if x_prod else "",
                "lags_dy_P": P,
                "lags_dx_prod_Q": Q_PROD if x_prod else 0,
                "lags_dx_pos_Q": Q_POS,
                "lags_dx_neg_Q": Q_NEG,
                "wald_long_run_stat": w_lr,
                "wald_long_run_p": p_lr,
                "wald_long_run_flag": lr_flag,
                "wald_short_run_stat": w_sr,
                "wald_short_run_p": p_sr,
                "wald_short_run_flag": sr_flag,
                "n_obs": int(model.nobs),
                "aic": float(model.aic),
                "bic": float(model.bic),
            }
        )

    out = pd.DataFrame(rows)
    out_path = OUT_DIR / "wald_asymmetry_tests.csv"
    out.to_csv(out_path, index=False)

    print(f"[SUCCESS] Wald tests saved to: {out_path}")
    print(out[["market", "variety", "grade", "wald_long_run_p", "wald_short_run_p"]])


if __name__ == "__main__":
    main()
