"""
Revision library for the NASCDI-NARDL paper (response to referee reports).

Differences from the submitted pipeline (src/models/run_nardl_batch_log.py):
  1. Lags and differences are taken over CALENDAR weeks (regular W-SUN grid). A row whose
     lag window crosses a missing week is dropped instead of silently differencing across the gap.
  2. NASCDI- is the Shin et al. (2014) NEGATIVE partial sum (cumulative sum of min(dNASCDI, 0)),
     so that beta+ = beta- is the symmetry null (the submitted code stored |decreases|, which made
     the Wald restriction P - N = 0 test the wrong hypothesis).
  3. Lag orders (p, q, r) are selected per equation by AIC subject to a Breusch-Godfrey check.
  4. Cointegration is assessed with the bounds F- and t-tests against finite-sample critical
     values simulated for each equation's own n and lag structure (Narayan 2005 style).
  5. Symmetry (long-run, short-run, joint) is tested with HAC Wald statistics and restricted
     wild-bootstrap p-values.
"""
from __future__ import annotations

import glob
import itertools
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import (
    acorr_breusch_godfrey,
    acorr_ljungbox,
    het_arch,
    het_breuschpagan,
    het_white,
    linear_reset,
)
from statsmodels.stats.stattools import jarque_bera

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "data" / "model_news"
NAS_WEEKLY = ROOT / "data" / "nascdi_news" / "nascdi_news_weekly.csv"
OUT_DIR = ROOT / "results_revision"
OUT_DIR.mkdir(exist_ok=True)

CHAIN_RE = re.compile(r"weekly_model_(.+)_(American|Delicious)_(A|B)_posneg\.csv$")
HAC_LAGS = 4


# ----------------------------------------------------------------------------- data
def load_nascdi(end: str | None = None) -> pd.DataFrame:
    w = pd.read_csv(NAS_WEEKLY, parse_dates=["week_end"]).sort_values("week_end").set_index("week_end")
    if end:
        w = w[w.index <= pd.Timestamp(end)]
    w["logvol"] = np.log1p(w["news_volume"])
    return w


def partial_sums(z: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Shin et al. (2014): P = cumsum(max(dz,0)), N = cumsum(min(dz,0)) <= 0."""
    dz = z.diff().fillna(0.0)
    return dz.clip(lower=0).cumsum(), dz.clip(upper=0).cumsum()


def chains() -> list[tuple[str, Path]]:
    out = []
    for f in sorted(glob.glob(str(MODEL_DIR / "weekly_model_*_posneg.csv"))):
        m = CHAIN_RE.search(os.path.basename(f))
        out.append((f"{m.group(1)}-{m.group(2)}-{m.group(3)}", Path(f)))
    return out


def chain_frame(path: Path, index: pd.Series, extra: dict[str, pd.Series] | None = None) -> pd.DataFrame:
    """Regular weekly grid for one chain. `index` is the weekly NASCDI-type series (continuous)."""
    x = pd.read_csv(path, parse_dates=["week_end"]).sort_values("week_end").set_index("week_end")
    grid = pd.date_range(x.index.min(), x.index.max(), freq="W-SUN")
    P, N = partial_sums(index)
    f = pd.DataFrame(index=grid)
    f["y"] = x["log_price_prod"].reindex(grid)
    f["x"] = x["log_price_term"].reindex(grid)
    f["pprod"] = x["avg_price_prod"].reindex(grid)
    f["Z"] = index.reindex(grid)
    f["P"] = P.reindex(grid)
    f["N"] = N.reindex(grid)
    if extra:
        for k, v in extra.items():
            f[k] = v.reindex(grid)
    return f


# ----------------------------------------------------------------------------- design
def build_design(f: pd.DataFrame, p: int, q: int, r: int, spec: str = "asym", dep: str = "prod",
                 controls: list[str] | None = None):
    """
    ECM design on the calendar-week grid.
      spec 'asym'   : levels y,x,P,N ; short-run dy(1..p), dx(0..q), dP,dN(0..r)
           'lr_sym' : levels y,x,Z   ; short-run dP,dN asymmetric
           'sr_sym' : levels y,x,P,N ; short-run dZ (symmetric)
           'sym'    : levels y,x,Z   ; short-run dZ
    Returns (d2, y, X) with rows containing NaN dropped (i.e. windows crossing a gap are removed).
    """
    yv, xv = (f["y"], f["x"]) if dep == "prod" else (f["x"], f["y"])
    d = pd.DataFrame(index=f.index)
    d["dy"] = yv.diff()
    d["y_L1"] = yv.shift(1)
    d["x_L1"] = xv.shift(1)
    dx = xv.diff()
    dz, dP, dN = f["Z"].diff(), f["P"].diff(), f["N"].diff()
    if spec in ("asym", "sr_sym"):
        d["P_L1"], d["N_L1"] = f["P"].shift(1), f["N"].shift(1)
    else:
        d["Z_L1"] = f["Z"].shift(1)
    for i in range(1, p + 1):
        d[f"dy_L{i}"] = d["dy"].shift(i)
    for j in range(0, q + 1):
        d[f"dx_L{j}"] = dx.shift(j)
    for k in range(0, r + 1):
        if spec in ("asym", "lr_sym"):
            d[f"dP_L{k}"] = dP.shift(k)
            d[f"dN_L{k}"] = dN.shift(k)
        else:
            d[f"dZ_L{k}"] = dz.shift(k)
    for c in controls or []:
        d[c] = f[c]
    d = d.dropna()
    y = d["dy"]
    X = sm.add_constant(d.drop(columns="dy"), has_constant="add")
    return d, y, X


def ols(y, X, cov="HAC"):
    if cov == "HAC":
        return sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS, "use_correction": True})
    if cov == "HC3":
        return sm.OLS(y, X).fit(cov_type="HC3")
    return sm.OLS(y, X).fit()


def half_life(ect: float) -> float:
    return np.log(0.5) / np.log(1.0 + ect) if -1 < ect < 0 else np.nan


# ----------------------------------------------------------------------------- tests
def wald(res, R: np.ndarray, q: np.ndarray | None = None):
    R = np.atleast_2d(R)
    q = np.zeros(R.shape[0]) if q is None else np.asarray(q, float)
    b = np.asarray(res.params)
    V = np.asarray(res.cov_params())
    d = R @ b - q
    W = float(d @ np.linalg.solve(R @ V @ R.T, d))
    return W, float(stats.chi2.sf(W, R.shape[0]))


def restr_matrices(names: list[str]):
    """Long-run (beta+ = beta-), short-run cumulative (sum theta+ = sum theta-), joint."""
    lr = np.zeros((1, len(names)))
    lr[0, names.index("P_L1")] = 1
    lr[0, names.index("N_L1")] = -1
    sr = np.zeros((1, len(names)))
    for k, n in enumerate(names):
        if n.startswith("dP_L"):
            sr[0, k] = 1
        if n.startswith("dN_L"):
            sr[0, k] = -1
    return lr, sr, np.vstack([lr, sr])


def rls(X: np.ndarray, y: np.ndarray, R: np.ndarray):
    XtXi = np.linalg.inv(X.T @ X)
    b = XtXi @ X.T @ y
    A = R @ XtXi @ R.T
    return b - XtXi @ R.T @ np.linalg.solve(A, R @ b)


def wild_boot_wald(y: pd.Series, X: pd.DataFrame, R: np.ndarray, B: int = 1999, seed: int = 1):
    """Restricted wild bootstrap (Rademacher) p-value for a linear restriction, HC3 Wald statistic."""
    Xv, yv = X.values, y.values
    n, k = Xv.shape
    XtXi = np.linalg.inv(Xv.T @ Xv)
    H = np.einsum("ij,jk,ik->i", Xv, XtXi, Xv)

    def hc3_wald(yy):
        b = XtXi @ Xv.T @ yy
        e = yy - Xv @ b
        u = e / (1 - H)
        meat = (Xv * u[:, None]).T @ (Xv * u[:, None])
        V = XtXi @ meat @ XtXi
        d = R @ b
        return float(d @ np.linalg.solve(R @ V @ R.T, d))

    W0 = hc3_wald(yv)
    br = rls(Xv, yv, R)
    fit_r = Xv @ br
    e_r = (yv - fit_r) / np.sqrt(1 - H)
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(B):
        v = rng.choice([-1.0, 1.0], size=n)
        if hc3_wald(fit_r + e_r * v) >= W0:
            cnt += 1
    return W0, (1 + cnt) / (B + 1)


def diagnostics(res_plain, d2: pd.DataFrame, X: pd.DataFrame):
    r = res_plain.resid
    out = {}
    out["LB_p8"] = float(acorr_ljungbox(r, lags=[min(8, len(r) // 4)], return_df=True)["lb_pvalue"].iloc[0])
    try:
        out["BG_p4"] = float(acorr_breusch_godfrey(res_plain, nlags=4)[1])
    except Exception:
        out["BG_p4"] = np.nan
    out["BP_p"] = float(het_breuschpagan(r, X.values)[1])
    try:
        out["White_p"] = float(het_white(r, X.values)[1])
    except Exception:
        out["White_p"] = np.nan
    try:
        out["ARCH_p4"] = float(het_arch(r, nlags=4)[1])
    except Exception:
        out["ARCH_p4"] = np.nan
    try:
        out["RESET_p"] = float(linear_reset(res_plain, power=3, test_type="fitted", use_f=True).pvalue)
    except Exception:
        out["RESET_p"] = np.nan
    out["JB_p"] = float(jarque_bera(r)[1])
    return out


# ----------------------------------------------------------------------------- bounds test
def bounds_stats(res_plain, names: list[str], spec: str):
    lev = ["y_L1", "x_L1"] + (["P_L1", "N_L1"] if spec in ("asym", "sr_sym") else ["Z_L1"])
    R = np.zeros((len(lev), len(names)))
    for i, n in enumerate(lev):
        R[i, names.index(n)] = 1
    F = float(res_plain.f_test(R).fvalue)
    t = float(res_plain.tvalues["y_L1"])
    return F, t, len(lev) - 1  # k = number of long-run regressors


def _sim_design(T, p, q, r, k, upper, rng):
    """Simulated driftless series; k long-run regressors (x plus k-1 others), same lag structure."""
    # Null of no level relationship: y is always I(1); the regressors are I(0) (lower bound) or I(1) (upper bound).
    shocks = rng.standard_normal((T, k + 1))
    y = np.cumsum(shocks[:, 0])
    X = np.cumsum(shocks[:, 1:], axis=0) if upper else shocks[:, 1:]
    dy = np.diff(y, prepend=np.nan)
    dX = np.diff(X, axis=0, prepend=np.full((1, k), np.nan))
    L = max(p, q, r) + 1
    cols = [np.ones(T)]
    lev_idx = []
    cols.append(np.r_[np.nan, y[:-1]]); lev_idx.append(len(cols) - 1)
    for j in range(k):
        cols.append(np.r_[np.nan, X[:-1, j]]); lev_idx.append(len(cols) - 1)
    for i in range(1, p + 1):
        cols.append(np.r_[np.full(i, np.nan), dy[:-i]])
    for j in range(0, q + 1):
        cols.append(np.r_[np.full(j, np.nan), dX[:T - j, 0]])
    for kk in range(0, r + 1):
        for jj in range(1, k):
            cols.append(np.r_[np.full(kk, np.nan), dX[:T - kk, jj]])
    M = np.column_stack(cols)
    ok = ~np.isnan(M).any(axis=1) & ~np.isnan(dy)
    return dy[ok], M[ok], lev_idx


def sim_bounds_cv(n: int, p: int, q: int, r: int, k: int, reps: int = 4000, seed: int = 7):
    """Finite-sample critical values for the F (all level terms) and t (y_L1) statistics, both bounds."""
    rng = np.random.default_rng(seed)
    res = {}
    for upper in (False, True):
        Fs, Ts = [], []
        for _ in range(reps):
            T = n + max(p, q, r) + 3
            dy, M, lev = _sim_design(T, p, q, r, k, upper, rng)
            dy, M = dy[-n:], M[-n:]
            kk = M.shape[1]
            b = np.linalg.lstsq(M, dy, rcond=None)[0]
            e = dy - M @ b
            rss_u = e @ e
            keep = [i for i in range(kk) if i not in lev]
            Mr = M[:, keep]
            br = np.linalg.lstsq(Mr, dy, rcond=None)[0]
            er = dy - Mr @ br
            rss_r = er @ er
            m = len(lev)
            Fs.append(((rss_r - rss_u) / m) / (rss_u / (n - kk)))
            XtXi = np.linalg.inv(M.T @ M)
            s2 = rss_u / (n - kk)
            Ts.append(b[lev[0]] / np.sqrt(s2 * XtXi[lev[0], lev[0]]))
        Fs, Ts = np.array(Fs), np.array(Ts)
        res["U" if upper else "L"] = {
            "F10": np.quantile(Fs, .90), "F05": np.quantile(Fs, .95), "F01": np.quantile(Fs, .99),
            "t10": np.quantile(Ts, .10), "t05": np.quantile(Ts, .05), "t01": np.quantile(Ts, .01),
        }
    return res


# ----------------------------------------------------------------------------- lag selection
def select_lags(f: pd.DataFrame, pmax=3, qmax=3, rmax=3, spec="asym", dep="prod", ic="aic",
                require_bg=True, controls=None):
    """Grid search on a COMMON sample (the sample available at the maximum lags)."""
    d_ref, _, _ = build_design(f, pmax, qmax, rmax, spec, dep, controls)
    common_idx = d_ref.index
    rows = []
    for p, q, r in itertools.product(range(pmax + 1), range(qmax + 1), range(rmax + 1)):
        d, y, X = build_design(f, p, q, r, spec, dep, controls)
        idx = common_idx.intersection(d.index)
        if len(idx) < len(common_idx):
            continue
        y, X = y.loc[common_idx], X.loc[common_idx]
        if X.shape[1] >= len(y) - 5:
            continue
        m = sm.OLS(y, X).fit()
        try:
            bg = float(acorr_breusch_godfrey(m, nlags=4)[1])
        except Exception:
            bg = np.nan
        rows.append(dict(p=p, q=q, r=r, aic=m.aic, bic=m.bic, bg_p=bg, n_common=len(y), k=X.shape[1]))
    g = pd.DataFrame(rows)
    key = ic
    ok = g[g.bg_p > 0.05] if require_bg else g
    flagged = False
    if ok.empty:
        ok, flagged = g, True
    best = ok.sort_values(key).iloc[0]
    return int(best.p), int(best.q), int(best.r), g, flagged


def fit_chain(f: pd.DataFrame, p: int, q: int, r: int, spec="asym", dep="prod", controls=None):
    d, y, X = build_design(f, p, q, r, spec, dep, controls)
    plain = ols(y, X, "plain")
    hac = ols(y, X, "HAC")
    return d, y, X, plain, hac
