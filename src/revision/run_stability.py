"""Referee 1 #4-#5: parameter stability of the error-correction relation and reliability of inference
under variance instability (wild-bootstrap sup-F, known-date Chow at the coverage break, bootstrap bounds F)."""
import numpy as np
import pandas as pd
from scipy import stats

from . import rev_lib as L

BREAK = pd.Timestamp("2018-03-04")
LEV = ["y_L1", "x_L1", "P_L1", "N_L1"]


def hc3_wald_block(y, X, cols):
    Xv = X.values
    XtXi = np.linalg.inv(Xv.T @ Xv)
    b = XtXi @ Xv.T @ y
    e = y - Xv @ b
    H = np.einsum("ij,jk,ik->i", Xv, XtXi, Xv)
    u = e / (1 - H)
    V = XtXi @ ((Xv * u[:, None]).T @ (Xv * u[:, None])) @ XtXi
    idx = [list(X.columns).index(c) for c in cols]
    d = b[idx]
    return float(d @ np.linalg.solve(V[np.ix_(idx, idx)], d))


def supwald(y, X, dates, cand_idx):
    best, argb = -1, None
    for i in cand_idx:
        D = (np.arange(len(y)) >= i).astype(float)
        Xi = X.copy()
        for c in LEV:
            Xi[c + "_D"] = X[c].values * D
        try:
            w = hc3_wald_block(y.values, Xi, [c + "_D" for c in LEV])
        except np.linalg.LinAlgError:
            continue
        if w > best:
            best, argb = w, i
    return best, argb


def boot_sup(y, X, cand_idx, B, rng):
    Xv = X.values
    XtXi = np.linalg.inv(Xv.T @ Xv)
    b = XtXi @ Xv.T @ y.values
    fit = Xv @ b
    H = np.einsum("ij,jk,ik->i", Xv, XtXi, Xv)
    e = (y.values - fit) / np.sqrt(1 - H)
    out = []
    for _ in range(B):
        yb = pd.Series(fit + e * rng.choice([-1.0, 1.0], size=len(e)), index=y.index)
        out.append(supwald(yb, X, None, cand_idx)[0])
    return np.array(out)


if __name__ == "__main__":
    base = pd.read_csv(L.OUT_DIR / "main_base.csv").set_index("chain")
    nas = L.load_nascdi()
    rng = np.random.default_rng(21)
    rows = []
    for name, path in L.chains():
        p, q, r = int(base.loc[name, "p"]), int(base.loc[name, "q"]), int(base.loc[name, "r"])
        f = L.chain_frame(path, nas["NASCDI"])
        d, y, X = L.build_design(f, p, q, r, "asym")
        n = len(y)
        lo, hi = int(0.2 * n), int(0.8 * n)
        cand = list(range(lo, hi))
        W, i_hat = supwald(y, X, d.index, cand)
        sims = boot_sup(y, X, cand, 299, rng)
        row = dict(chain=name, n=n, supWald=W, sup_date=str(d.index[i_hat].date()), supWald_p_boot=(1 + (sims >= W).sum()) / 300)
        # known-date test at the coverage break (only if the break lies inside the estimation sample with enough rows each side)
        k0 = int(np.searchsorted(d.index, BREAK))
        if 0.15 * n < k0 < 0.85 * n:
            w0, _ = supwald(y, X, d.index, [k0])
            sims0 = boot_sup(y, X, [k0], 299, rng)
            row.update(chow_W=w0, chow_p_boot=(1 + (sims0 >= w0).sum()) / 300, n_pre=k0, n_post=n - k0)
        # variance instability: residual variance before / after the coverage break (Goldfeld-Quandt style)
        plain = L.ols(y, X, "plain")
        res = pd.Series(plain.resid.values, index=d.index)
        a, b = res[res.index < BREAK], res[res.index >= BREAK]
        if len(a) > 8 and len(b) > 8:
            Fv = b.var(ddof=1) / a.var(ddof=1)
            row.update(var_ratio_post_pre=Fv, var_ratio_p=2 * min(stats.f.sf(Fv, len(b) - 1, len(a) - 1), stats.f.cdf(Fv, len(b) - 1, len(a) - 1)))
        # bootstrap bounds-F (heteroskedasticity robust, regressors held at their sample values)
        R = np.zeros((4, X.shape[1]))
        for i, c in enumerate(LEV):
            R[i, list(X.columns).index(c)] = 1
        Wb, pb = L.wild_boot_wald(y, X, R, B=1999, seed=9)
        row.update(levels_wald_hc3=Wb, levels_p_wildboot=pb)
        # ECT inference under alternative covariance estimators
        for tag in ("plain", "HC3", "HAC"):
            m = L.ols(y, X, tag)
            row[f"ECT_se_{tag}"] = m.bse["y_L1"]
        rows.append(row)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    pd.DataFrame(rows).to_csv(L.OUT_DIR / "stability_and_inference.csv", index=False)
