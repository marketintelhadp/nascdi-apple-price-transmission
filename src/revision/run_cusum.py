"""CUSUM / CUSUMSQ for the revised (calendar-week, selected-lag) unrestricted NARDL equations."""
import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import breaks_cusumolsresid

from . import rev_lib as L


def recursive_resid(y: np.ndarray, X: np.ndarray):
    n, k = X.shape
    w = []
    for t in range(k, n):
        Xt = X[:t]
        XtXi = np.linalg.pinv(Xt.T @ Xt)
        b = XtXi @ Xt.T @ y[:t]
        x = X[t]
        w.append((y[t] - x @ b) / np.sqrt(1 + x @ XtXi @ x))
    return np.array(w)


if __name__ == "__main__":
    base = pd.read_csv(L.OUT_DIR / "main_base.csv").set_index("chain")
    nas = L.load_nascdi()
    rows = []
    for name, path in L.chains():
        p, q, r = int(base.loc[name, "p"]), int(base.loc[name, "q"]), int(base.loc[name, "r"])
        f = L.chain_frame(path, nas["NASCDI"])
        d, y, X = L.build_design(f, p, q, r, "asym")
        res = L.ols(y, X, "plain")
        k = X.shape[1]
        stat, pv, _ = breaks_cusumolsresid(res.resid.values, ddof=k)
        w = recursive_resid(y.values, X.values)
        T = len(w)
        s = np.cumsum(w ** 2) / np.sum(w ** 2)
        grid = np.arange(1, T + 1) / T
        c0 = 1.358 / np.sqrt(T / 2.0)  # large-sample 5% band half-width for CUSUMSQ (Brown-Durbin-Evans approximation)
        breach = bool(np.any(np.abs(s - grid) > c0))
        sig = np.sqrt(np.sum(w ** 2) / T)
        W = np.cumsum(w) / sig
        a = 0.948
        bound = a * np.sqrt(T) * (1 + 2 * np.arange(1, T + 1) / T)
        rows.append(dict(chain=name, n=len(y), cusum_stat=stat, cusum_p=pv, cusum_breach_BDE=bool(np.any(np.abs(W) > bound)),
                         cusumsq_max_dev=float(np.max(np.abs(s - grid))), cusumsq_band=c0, cusumsq_breach=breach))
    out = pd.DataFrame(rows)
    out.to_csv(L.OUT_DIR / "cusum_revised.csv", index=False)
    print(out.round(3).to_string())
