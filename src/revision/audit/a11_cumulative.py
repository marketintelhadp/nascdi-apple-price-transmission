"""Cumulative short-run effect of a +1 SD (5.6 points) sustained 5-week NASCDI increase on dlog(price), with wild-bootstrap p-values."""
import importlib.util, os, numpy as np, pandas as pd, statsmodels.api as sm
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("rev_lib", os.path.join(HERE, "..", "rev_lib.py"))
L = importlib.util.module_from_spec(spec); spec.loader.exec_module(L)
nas = L.load_nascdi(); rows = []
for direction in ("producer", "terminal"):
    for name, path in L.chains():
        f = L.chain_frame(path, nas["NASCDI"])
        if direction == "terminal": f = f.copy(); f["y"], f["x"] = f["x"].copy(), f["y"].copy()
        d = pd.DataFrame(index=f.index); d["dy"] = f.y.diff(); dx = f.x.diff(); dz = f.Z.diff()
        d["dy_L1"] = d.dy.shift(1)
        for j in range(3): d[f"dx_L{j}"] = dx.shift(j)
        for k in range(5): d[f"dZ_L{k}"] = dz.shift(k)
        d = d.dropna(); X = sm.add_constant(d.drop(columns="dy")); y = d.dy; names = list(X.columns)
        R = np.zeros((1, len(names)))
        for i, n in enumerate(names):
            if n.startswith("dZ_L"): R[0, i] = 1
        m = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
        est = float((R @ m.params.values)[0]); se = float(np.sqrt((R @ m.cov_params().values @ R.T)[0, 0]))
        W, pb = L.wild_boot_wald(y, X, R, B=1999, seed=9)
        rows.append(dict(direction=direction, chain=name, n=len(y), cum_pct=est * 5.6 * 100, se_hac_pct=se * 5.6 * 100, p_hac=float(m.pvalues.iloc[0]) if False else float(2 * (1 - __import__('scipy').stats.norm.cdf(abs(est / se)))), p_boot=pb))
        print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(os.path.join(HERE, "..", "..", "..", "results_revision", "audit", "nascdi_cumulative_effect.csv"), index=False)
