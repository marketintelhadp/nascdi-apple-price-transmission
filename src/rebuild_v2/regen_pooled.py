"""Earlier-design pooled test on the rebuilt index: eight producer-price equations stacked, chain fixed effects (chain-specific intercepts),
chain-specific lagged levels and own-lag, COMMON coefficients on five short-run index terms (change in NASCDI, lags 0-4); joint test that the five are zero.
Inference: week-clustered Wald (CR1) and restricted wild bootstrap with a common week-level Rademacher multiplier (B=1999)."""
import numpy as np, pandas as pd, sys
from regen import CH, CHAINS, IDX, rng
IDXD = IDX["NASCDI"].diff()
def design(ch, nl=5, extra_dy=1):
    df = CH[ch].set_index("week_end"); grid = pd.date_range(df.index.min(), df.index.max(), freq="W-SUN")
    y = df["log_price_prod"].reindex(grid); x = df["log_price_term"].reindex(grid); a = IDXD.reindex(grid)
    # the index change must be observed over the true calendar week; IDX is a continuous weekly series
    d = pd.DataFrame({"dy": y.diff(), "y_L1": y.shift(1), "x_L1": x.shift(1), "dx": x.diff()}, index=grid)
    for i in range(1, extra_dy + 1): d[f"dy_L{i}"] = y.diff().shift(i)
    for i in range(nl): d[f"da_L{i}"] = IDX["NASCDI"].diff().reindex(grid.union(grid - pd.Timedelta(days=7*nl))).reindex(grid).shift(0) if False else IDX["NASCDI"].diff().shift(i).reindex(grid)
    d["chain"] = ch; return d.dropna()
def blocks(spec):
    rows = []
    for ch in CHAINS:
        d = design(ch, extra_dy=spec["dy"]); rows.append(d)
    D = pd.concat(rows); cols_ch = ["const", "y_L1", "x_L1", "dx"] + [f"dy_L{i}" for i in range(1, spec["dy"] + 1)]
    Xs = []
    for ch in CHAINS:
        m = (D.chain == ch).values; B = np.zeros((len(D), len(cols_ch)))
        B[m, 0] = 1
        for j, c in enumerate(cols_ch[1:], 1): B[m, j] = D.loc[m, c].values
        Xs.append(B)
    Xc = np.hstack(Xs); Xa = D[[f"da_L{i}" for i in range(5)]].values
    return D, np.hstack([Xc, Xa]), Xc.shape[1]
def cr_wald(y, X, R, groups):
    XtXi = np.linalg.inv(X.T @ X); b = XtXi @ X.T @ y; e = y - X @ b; k = X.shape[1]; G = len(np.unique(groups)); n = len(y)
    S = np.zeros((k, k))
    for g in np.unique(groups): m = groups == g; s = X[m].T @ e[m]; S += np.outer(s, s)
    V = XtXi @ S @ XtXi * (G / (G - 1)) * ((n - 1) / (n - k)); r = R @ b
    return float(r @ np.linalg.solve(R @ V @ R.T, r)), b
res = []
for lab, spec in (("own lag 1", dict(dy=1)), ("no own lag", dict(dy=0)), ("own lags 1-2", dict(dy=2))):
    D, X, nb = blocks(spec); y = D["dy"].values; groups = D.index.values.astype("datetime64[ns]").astype(np.int64)
    R = np.zeros((5, X.shape[1])); R[np.arange(5), nb + np.arange(5)] = 1
    obs, b = cr_wald(y, X, R, groups); XtXi = np.linalg.inv(X.T @ X); bc = b - XtXi @ R.T @ np.linalg.solve(R @ XtXi @ R.T, R @ b); f0 = X @ bc; e0 = y - f0
    weeks = np.unique(groups); cnt = 0; B = 1999
    for _ in range(B):
        mult = dict(zip(weeks, rng.choice([-1.0, 1.0], size=len(weeks)))); w = np.array([mult[g] for g in groups]); cnt += cr_wald(f0 + e0 * w, X, R, groups)[0] >= obs
    from scipy import stats
    res.append(dict(spec=lab, n=len(D), weeks=len(weeks), wald=obs, p_chi2=stats.chi2.sf(obs, 5), p_wildboot=(cnt + 1) / (B + 1), coefs=np.round(b[nb:nb + 5], 5).tolist(), cum_sum=float(b[nb:nb + 5].sum())))
    print(res[-1], flush=True)
pd.DataFrame(res).to_csv(TAB / "tableA11_pooled_earlier_design.csv", index=False)
