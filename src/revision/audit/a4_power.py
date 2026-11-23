import importlib.util, numpy as np, pandas as pd, statsmodels.api as sm
spec = importlib.util.spec_from_file_location("rev_lib", __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "rev_lib.py"))
L = importlib.util.module_from_spec(spec); spec.loader.exec_module(L)
nas = L.load_nascdi(); Z = nas["NASCDI"]; sdZ = Z.diff().std()
print("sd dNASCDI", round(sdZ, 2))

# ---- A. minimal detectable cumulative effect of NASCDI (short-run, 5 weekly lags) per chain + pooled panel
frames = []; mde_rows = []
for name, path in L.chains():
    f = L.chain_frame(path, Z)
    d = pd.DataFrame(index=f.index)
    d["dy"] = f.y.diff(); dx = f.x.diff(); dz = f.Z.diff()
    d["dy_L1"] = d.dy.shift(1)
    for j in range(3): d[f"dx_L{j}"] = dx.shift(j)
    for k in range(5): d[f"dZ_L{k}"] = dz.shift(k)
    d = d.dropna(); d["chain"] = name; frames.append(d)
    X = sm.add_constant(d.drop(columns=["dy", "chain"])); m = sm.OLS(d.dy, X).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
    zc = [c for c in X.columns if c.startswith("dZ_L")]
    R = np.zeros((1, X.shape[1])); R[0, [list(X.columns).index(c) for c in zc]] = 1
    est = float((R @ m.params.values)[0]); se = float(np.sqrt((R @ m.cov_params().values @ R.T)[0,0]))
    mde_rows.append(dict(chain=name, n=len(d), cum_pct=est*5.6*100, se_pct=se*5.6*100, mde80_pct=2.8*se*5.6*100))
    print(f"{name:20s} n={len(d):3d} cum effect of +1SD(5.6) sustained 5wk: {est*5.6*100:6.2f}%  se {se*5.6*100:5.2f}%  MDE80 {2.8*se*5.6*100:5.2f}%")
D = pd.concat(frames); D["week"] = D.index
Xp = pd.get_dummies(D["chain"], dtype=float)
cols = [c for c in D.columns if c.startswith("dx_L") or c.startswith("dZ_L") or c == "dy_L1"]
XP = pd.concat([Xp, D[cols]], axis=1)
mp = sm.OLS(D.dy, XP).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(D.week)[0]})
zc = [c for c in cols if c.startswith("dZ_L")]
R = np.zeros((len(zc), XP.shape[1]))
for i, c in enumerate(zc): R[i, list(XP.columns).index(c)] = 1
w = mp.wald_test(R, scalar=True)
import os; _o = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "results_revision", "audit")
pd.DataFrame(mde_rows).to_csv(os.path.join(_o, "nascdi_detectable_effect.csv"), index=False)
pd.DataFrame([dict(pooled_p=float(w.pvalue), cum_pct_per_SD=float(R.sum(0) @ mp.params.values) * 5.6 * 100)]).to_csv(os.path.join(_o, "nascdi_pooled.csv"), index=False)
print("POOLED (8 chains, chain FE, week-clustered): joint dZ0..4 chi2 p =", round(float(w.pvalue), 3), "| cum effect %.2f%% per +1SD" % (float(R.sum(0) @ mp.params.values) * 5.6 * 100))

# ---- B. power of spread / bounds-type t test, calibrated to Sopore-Delicious-A-like and Shopian chains
def sim_power(n, alpha, reps=3000, seed=1, sx=0.05, sy=0.06, rho=0.4, gam=0.3):
    rng = np.random.default_rng(seed)
    def stat(a):
        t = np.empty(reps)
        for i in range(reps):
            e = rng.standard_normal(n + 50) * sy; ux = rng.standard_normal(n + 50) * sx
            x = np.cumsum(ux); y = np.zeros(n + 50)
            for k in range(1, n + 50): y[k] = y[k-1] + a * (y[k-1] - x[k-1]) + gam * ux[k] + e[k]
            y, x = y[50:], x[50:]
            dy = np.diff(y); s = (y - x)[:-1]; dx = np.diff(x)
            X = np.column_stack([np.ones(n - 1), s, dx])
            b = np.linalg.lstsq(X, dy, rcond=None)[0]; r = dy - X @ b
            se = np.sqrt(r @ r / (n - 1 - 3) * np.linalg.inv(X.T @ X)[1, 1]); t[i] = b[1] / se
        return t
    crit = np.quantile(stat(0.0), .05)
    return crit, float((stat(alpha) < crit).mean())
for n in (53, 100, 200, 230):
    out = [f"n={n}"]
    for a in (-0.05, -0.10, -0.15):
        c, pw = sim_power(n, a, reps=1500); out.append(f"alpha={a}: crit {c:.2f} power {pw:.2f}")
    print(" | ".join(out), flush=True)
