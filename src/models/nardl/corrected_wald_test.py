# ============================================================
# Wald Tests for NARDL Asymmetry (ALIGNED WITH ECM DESIGN)
# - Uses SAME design matrix as estimation (build_ecm_design)
# - Ensures identical sample size (n_obs) as Table 5
# ============================================================

from __future__ import annotations

import re
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

# -----------------------------
# CONFIG (MUST MATCH ESTIMATION)
# -----------------------------
MAX_LAGS_Y = 4
MAX_LAGS_X = 4

# -----------------------------
# Repo root finder
# -----------------------------
def find_repo_root(start: Path) -> Path:
    cur = start.resolve()
    for _ in range(10):
        if (cur / "src").exists() and (cur / "data").exists():
            return cur
        cur = cur.parent
    return start.resolve().parents[3]

# -----------------------------
# ECM DESIGN (IDENTICAL TO TABLE 5)
# -----------------------------
def build_ecm_design(df: pd.DataFrame):

    d = df.sort_values("week_end").copy()

    y = "log_price_term" if "log_price_term" in d.columns else "avg_price_term"
    x = "log_price_prod" if "log_price_prod" in d.columns else "avg_price_prod"
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

    X = sm.add_constant(d2[Xcols], has_constant="add")

    return d2, Y, X, Xcols

# -----------------------------
# WALD TESTS
# -----------------------------
def wald_long_run(model):
    """
    H0: p_L1 = n_L1 (long-run symmetry)
    """
    params = model.params.index.tolist()

    if "p_L1" not in params or "n_L1" not in params:
        return np.nan, np.nan

    R = np.zeros((1, len(params)))
    R[0, params.index("p_L1")] = 1
    R[0, params.index("n_L1")] = -1

    test = model.wald_test(R)
    return float(test.statistic), float(test.pvalue)

def wald_short_run(model):
    """
    H0: sum(dp_L*) = sum(dn_L*) (short-run symmetry)
    """
    params = model.params.index.tolist()

    dp_terms = [p for p in params if p.startswith("dp_L")]
    dn_terms = [p for p in params if p.startswith("dn_L")]

    if not dp_terms or not dn_terms:
        return np.nan, np.nan

    R = np.zeros((1, len(params)))

    for t in dp_terms:
        R[0, params.index(t)] += 1

    for t in dn_terms:
        R[0, params.index(t)] -= 1

    test = model.wald_test(R)
    return float(test.statistic), float(test.pvalue)

# -----------------------------
# MAIN
# -----------------------------
def main():

    ROOT = find_repo_root(Path(__file__))
    DATA_DIR = ROOT / "data" / "model"
    OUT_DIR = ROOT / "results" / "nardl" / "tables"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(DATA_DIR.glob("weekly_model_*_posneg.csv"))

    rows = []

    for f in files:

        df = pd.read_csv(f, parse_dates=["week_end"])

        d2, Y, X, Xcols = build_ecm_design(df)

        model = sm.OLS(Y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 4})

        w_lr, p_lr = wald_long_run(model)
        w_sr, p_sr = wald_short_run(model)

        parts = f.stem.replace("weekly_model_", "").replace("_posneg", "").split("_")

        market, variety, grade = parts[0], parts[1], parts[2]

        rows.append({
            "market": market,
            "variety": variety,
            "grade": grade,
            "wald_long_run_stat": w_lr,
            "wald_long_run_p": p_lr,
            "wald_short_run_stat": w_sr,
            "wald_short_run_p": p_sr,
            "n_obs": int(len(d2)),  # SAME AS TABLE 5 NOW
            "file": f.name
        })

    out = pd.DataFrame(rows).sort_values(["market", "variety", "grade"])

    out_path = OUT_DIR / "wald_asymmetry_tests_fixed.csv"
    out.to_csv(out_path, index=False)

    print("✅ Wald tests FIXED and aligned.")
    print("Saved:", out_path)

if __name__ == "__main__":
    main()