"""Toda-Yamamoto (1995) Granger non-causality tests on the calendar-week grid (d_max = 1)."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from . import rev_lib as L


def lagmat(s: pd.Series, k: int, name: str):
    return pd.concat({f"{name}_L{i}": s.shift(i) for i in range(1, k + 1)}, axis=1)


def ty(f: pd.DataFrame, kmax=4, dmax=1):
    y, x = f["y"], f["x"]
    # lag order by AIC on a common sample (rows complete up to kmax + dmax lags)
    best = None
    ref = pd.concat([y, lagmat(y, kmax + dmax, "y"), lagmat(x, kmax + dmax, "x")], axis=1).dropna().index
    for k in range(1, kmax + 1):
        aic = 0.0
        for dep, oth, dn, on in ((y, x, "y", "x"), (x, y, "x", "y")):
            X = sm.add_constant(pd.concat([lagmat(dep, k + dmax, dn), lagmat(oth, k + dmax, on)], axis=1).loc[ref])
            m = sm.OLS(dep.loc[ref], X).fit()
            aic += m.aic
        if best is None or aic < best[0]:
            best = (aic, k)
    k = best[1]
    out = {"k": k}
    for tag, dep, oth, dn, on in (("term_to_prod", y, x, "y", "x"), ("prod_to_term", x, y, "x", "y")):
        D = pd.concat([dep, lagmat(dep, k + dmax, dn), lagmat(oth, k + dmax, on)], axis=1).dropna()
        X = sm.add_constant(D.drop(columns=dn if dn in D else D.columns[0]))
        yv = D.iloc[:, 0]
        X = sm.add_constant(D.iloc[:, 1:])
        m = sm.OLS(yv, X).fit()
        R = np.zeros((k, X.shape[1]))
        for i in range(1, k + 1):
            R[i - 1, list(X.columns).index(f"{on}_L{i}")] = 1
        Wd = float(m.wald_test(R, scalar=True).statistic)
        out[f"{tag}_chi2"], out[f"{tag}_p"], out["n"] = Wd, float(stats.chi2.sf(Wd, k)), len(D)
    return out


if __name__ == "__main__":
    nas = L.load_nascdi()
    rows = []
    for name, path in L.chains():
        f = L.chain_frame(path, nas["NASCDI"])
        r = ty(f)
        r["chain"] = name
        rows.append(r)
    out = pd.DataFrame(rows)[["chain", "n", "k", "term_to_prod_chi2", "term_to_prod_p", "prod_to_term_chi2", "prod_to_term_p"]]
    out.to_csv(L.OUT_DIR / "toda_yamamoto.csv", index=False)
    print(out.round(3).to_string())
