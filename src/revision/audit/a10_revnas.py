import importlib.util, numpy as np, pandas as pd, statsmodels.api as sm
src = open(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)),"a3_nascdi.py")).read().split("main = pd.read_csv")[0]
exec(src)
import itertools
rows = []
for name, path in L.chains():
    f = L.chain_frame(path, nas["NASCDI"]); fr = f.copy(); fr["y"], fr["x"] = f["x"], f["y"]   # terminal price dependent
    # short-run: d(terminal) on d(producer) lags and dZ lags
    n3, W3, p3, cum = shortrun(fr)
    n1, k1, W1, p1 = exclusion(fr, 0, 1, 1, "sym", "sym")
    n2, k2, W2, p2 = exclusion(fr, 0, 1, 1, "asym", "asym")
    rows.append(dict(chain=name, n_sr=n3, p_sr_dZ=round(p3,3), cum_pct_per_SD=round(cum*5.6*100,2), p_excl_sym=round(p1,3), p_excl_asym=round(p2,3)))
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "..", "..", "results_revision", "audit", "nascdi_terminal_equation.csv"), index=False)
