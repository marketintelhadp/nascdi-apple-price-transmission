# ============================================================
# CUSUM & CUSUMSQ stability tests for NARDL/ECM regressions
# - CUSUM uses statsmodels recursive_olsresiduals with 95% bands (5% sig.)
# - CUSUMSQ uses Monte-Carlo 95% envelope (robust + reviewer-proof)
# Outputs:
#   results/nardl/stability/cusum_*.png
#   results/nardl/stability/cusumsq_*.png
#   results/nardl/stability/cusum_cusumsq_summary.csv
# ============================================================

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from statsmodels.stats.diagnostic import (
    recursive_olsresiduals,
    breaks_cusumolsresid,
)

def choose_skip(exog: np.ndarray, min_skip: int = None) -> int:
    """
    Choose the smallest skip such that exog[:skip] has full column rank.
    Works around statsmodels recursive_olsresiduals 'singular' error.
    """
    nobs, k = exog.shape
    if min_skip is None:
        # at least enough observations to estimate all parameters + a buffer
        min_skip = k + 5

    # keep skip within sensible bounds
    start = max(min_skip, int(0.15 * nobs))  # 15% warm-up is a good default
    start = min(start, nobs - 5)

    for skip in range(start, nobs - 2):
        X0 = exog[:skip]
        if np.linalg.matrix_rank(X0) == k:
            return skip

    # fallback: if never full rank, return a conservative window
    return min(max(k + 5, int(0.30 * nobs)), nobs - 5)

# -----------------------
# CONFIG (adjust if needed)
# -----------------------
MAX_LAGS_Y = 4
MAX_LAGS_X = 4
MIN_OBS_AFTER_LAGS = 25

MC_DRAWS = 4000

# IMPORTANT:
# statsmodels recursive_olsresiduals expects alpha as CONFIDENCE LEVEL:
# allowed: 0.90, 0.95, 0.99
CONF_LEVEL = 0.95              # 95% bands = 5% significance
SIG_LEVEL = 1.0 - CONF_LEVEL   # 0.05

# -----------------------
# Paths (robust from file location)
# -----------------------
THIS = Path(__file__).resolve()
ROOT = THIS.parents[3]  # .../src/models/nardl -> repo root

DATA_DIR = ROOT / "data" / "model"
OUT_DIR = ROOT / "results" / "nardl" / "stability"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------
# Helpers
# -----------------------
def parse_series_name(fp: Path) -> tuple[str, str, str]:
    """
    weekly_model_<Market>_<Variety>_<Grade>_posneg.csv
    Market can have underscores if you previously sanitized spaces.
    """
    name = fp.name.replace("_posneg.csv", "")
    m = re.match(r"weekly_model_(.+)_(American|Delicious)_(A|B)$", name)
    if not m:
        return ("Unknown", "Unknown", "Unknown")
    market = m.group(1).replace("_", " ")
    return market, m.group(2), m.group(3)


def build_ecm_design(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    ECM/NARDL-style design (same structure as your batch script):
      dy = c + a*y_{t-1} + b*x_{t-1} + c1*p_{t-1} + c2*n_{t-1}
           + Σ dy_lags + Σ dx_lags + Σ dp_lags + Σ dn_lags + e
    """
    d = df.sort_values("week_end").copy()

    y = "avg_price_term"
    x = "avg_price_prod"
    p = "NASCDI_pos"
    n = "NASCDI_neg"

    req = ["week_end", y, x, p, n]
    miss = [c for c in req if c not in d.columns]
    if miss:
        raise ValueError(f"Missing columns: {miss}")

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


def mc_cusumsq_envelope(m: int, draws: int = MC_DRAWS, sig: float = SIG_LEVEL, seed: int = 123) -> tuple[np.ndarray, np.ndarray]:
    """
    Monte Carlo envelope for CUSUMSQ statistic:
      S_t = sum_{i<=t} e_i^2 / sum_{i<=m} e_i^2
    Under null (iid), approximate distribution via simulated N(0,1).
    Returns pointwise two-sided (1-sig) envelope.
    """
    rng = np.random.default_rng(seed + m)
    eps = rng.standard_normal(size=(draws, m))
    cs = np.cumsum(eps**2, axis=1)
    cs = cs / cs[:, [-1]]
    lo = np.quantile(cs, sig / 2.0, axis=0)
    hi = np.quantile(cs, 1.0 - sig / 2.0, axis=0)
    return lo, hi


def safe_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", s).strip("_")


# -----------------------
# Main
# -----------------------
def main():
    files = sorted(DATA_DIR.glob("weekly_model_*_posneg.csv"))
    if not files:
        raise FileNotFoundError(f"No files found in {DATA_DIR} matching weekly_model_*_posneg.csv")

    summary_rows = []

    for fp in files:
        market, variety, grade = parse_series_name(fp)

        df = pd.read_csv(fp)
        if "week_end" not in df.columns:
            raise ValueError(f"{fp.name} missing 'week_end' column")
        df["week_end"] = pd.to_datetime(df["week_end"], errors="coerce")

        df = df.dropna(subset=["week_end", "avg_price_prod", "avg_price_term", "NASCDI_pos", "NASCDI_neg"]).copy()

        try:
            d2, Y, X = build_ecm_design(df)
        except Exception as e:
            print(f"SKIP {fp.name} (design error: {e})")
            continue

        if len(d2) < MIN_OBS_AFTER_LAGS:
            print(f"SKIP {fp.name} (too short after lags: {len(d2)})")
            continue

        model = OLS(Y, X).fit()

        exog = model.model.exog
        skip = choose_skip(exog)

        out = recursive_olsresiduals(model, alpha=CONF_LEVEL, skip=skip)

        # statsmodels versions differ; keep first 6 outputs
        rresid, rparams, rypred, rresid_std, rcusum, rcusum_ci = out[:6]

        try:
            exog = model.model.exog
            skip = choose_skip(exog)
            out = recursive_olsresiduals(model, alpha=CONF_LEVEL, skip=skip)
            rresid, rparams, rypred, rresid_std, rcusum, rcusum_ci = out[:6]
        except Exception as e:
            print(f"SKIP {os.path.basename(fp)} -> CUSUM failed: {e}")
            continue



        # Formal CUSUM test p-value
        cusum_stat, cusum_p, cusum_crit = breaks_cusumolsresid(model.resid, ddof=model.df_model)

        # ---------- CUSUMSQ (Monte-Carlo 95% envelope)
        rr = np.asarray(rresid_std).astype(float)
        rr = rr[np.isfinite(rr)]
        m = len(rr)

        if m < 20:
            print(f"SKIP {fp.name} (CUSUMSQ too short: m={m})")
            continue

        cusumsq = np.cumsum(rr**2)
        cusumsq = cusumsq / cusumsq[-1]

        lo, hi = mc_cusumsq_envelope(m=m, draws=MC_DRAWS, sig=SIG_LEVEL, seed=123)
        cusumsq_outside = bool(np.any((cusumsq < lo) | (cusumsq > hi)))

        label = f"{market} | {variety} | G{grade}"
        tag = safe_filename(f"{market}_{variety}_{grade}")

        # ---------- Plot CUSUM
        fig = plt.figure(figsize=(11, 4))
        plt.plot(rcusum, linewidth=1.5)
        plt.plot(rcusum_ci[0], linestyle="--", linewidth=1.0)
        plt.plot(rcusum_ci[1], linestyle="--", linewidth=1.0)
        plt.axhline(0, linestyle=":", linewidth=1.0)
        plt.title(f"CUSUM Stability Test (95% bands): {label}")
        plt.xlabel("Recursive step")
        plt.ylabel("CUSUM")
        plt.tight_layout()
        cusum_path = OUT_DIR / f"cusum_{tag}.png"
        plt.savefig(cusum_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        # ---------- Plot CUSUMSQ
        fig = plt.figure(figsize=(11, 4))
        plt.plot(cusumsq, linewidth=1.8, label="CUSUMSQ")
        plt.plot(lo, linestyle="--", linewidth=1.0, label="95% envelope")
        plt.plot(hi, linestyle="--", linewidth=1.0)
        plt.ylim(-0.05, 1.05)
        plt.title(f"CUSUMSQ Stability Test (MC 95% envelope): {label}")
        plt.xlabel("Recursive step")
        plt.ylabel("Cumulative sum of squares (normalized)")
        plt.tight_layout()
        cusumsq_path = OUT_DIR / f"cusumsq_{tag}.png"
        plt.savefig(cusumsq_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        summary_rows.append({
            "producer_market": market,
            "variety": variety,
            "grade": grade,
            "n_obs_after_lags": int(len(d2)),
            "cusum_stat": float(np.asarray(cusum_stat).squeeze()),
            "cusum_pvalue": float(np.asarray(cusum_p).squeeze()),
            "cusumsq_outside_95_envelope": int(cusumsq_outside),
            "cusum_plot": str(cusum_path.relative_to(ROOT)),
            "cusumsq_plot": str(cusumsq_path.relative_to(ROOT)),
        })

        print(f"OK  {fp.name} -> CUSUM p={float(np.asarray(cusum_p).squeeze()):.4f} | CUSUMSQ outside={cusumsq_outside}")

    if not summary_rows:
        raise RuntimeError("No stability outputs were produced. Check your model files and MIN_OBS settings.")

    out_csv = OUT_DIR / "cusum_cusumsq_summary.csv"
    pd.DataFrame(summary_rows).sort_values(["producer_market", "variety", "grade"]).to_csv(out_csv, index=False)

    print("\n✅ Done.")
    print(f"Saved summary: {out_csv}")
    print(f"Saved plots in: {OUT_DIR}")


if __name__ == "__main__":
    main()
