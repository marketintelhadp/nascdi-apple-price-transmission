import importlib.util, numpy as np, pandas as pd, statsmodels.api as sm
exec(open(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)),"a4_power.py")).read().split("# ---- B.")[0].split("# ---- A.")[0])   # imports + L + nas
src = open(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)),"a4_power.py")).read()
exec("def sim_power" + src.split("def sim_power")[1].split("for n in (53")[0])
sel = {"Shopian-American-A": 53, "Shopian-American-B": 53, "Shopian-Delicious-A": 101, "Shopian-Delicious-B": 87, "Sopore-American-A": 187, "Sopore-American-B": 196, "Sopore-Delicious-A": 231, "Sopore-Delicious-B": 232}
rowsP = []
for name, path in L.chains():
    if name not in sel: continue
    f = L.chain_frame(path, Z); s = f.y - f.x
    d = pd.DataFrame({"dy": f.y.diff(), "s": s.shift(1), "dx": f.x.diff()}).dropna()
    X = sm.add_constant(d[["s", "dx"]]); m = sm.OLS(d.dy, X).fit()
    sx, sy, gam = d.dx.std(), m.resid.std(), m.params["dx"]
    res = []; rec = dict(chain=name, n=sel[name], sd_dx=sx, resid_sd=sy, gamma=gam)
    for a in (-0.05, -0.10, -0.15):
        c, pw = sim_power(sel[name], a, reps=1500, sx=sx, sy=sy, gam=gam); res.append(f"a={a}: power {pw:.2f}"); rec[f"power_{abs(a):.2f}"] = pw
    rowsP.append(rec)
    print(f"{name:20s} n={sel[name]} sdDx={sx:.3f} resid sd={sy:.3f} gam={gam:.2f} | " + " | ".join(res), flush=True)

import os; pd.DataFrame(rowsP).to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'results_revision', 'audit', 'power_cointegration.csv'), index=False)
