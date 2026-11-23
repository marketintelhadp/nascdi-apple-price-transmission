import sys, time
import numpy as np, pandas as pd, statsmodels.api as sm
from regen import *
t0 = time.time(); H = 26
pct = pd.read_csv(TAB / "tableA8b_percentiles.csv").set_index("percentile")["increase"]; sd = IDX["NASCDI"].diff().std()
shocks = {"1 SD": sd, "90th pct": pct[90], "99th pct": pct[99]}

def irf(params, names, p, r, s, h):
    """response of producer log price to an index spike of size s lasting h weeks (terminal price held at baseline)."""
    g = lambda nm: params[names.index(nm)] if nm in names else 0.0
    lam, gam = g("y_L1"), g("a_L1"); phi = [g(f"dy_L{i}") for i in range(1, p + 1)]; psi = [g(f"da_L{i}") for i in range(r + 1)]
    T = H + 1; a = np.zeros(T + 1); a[1:h + 1] = s     # a[t], t=0 is baseline
    y = np.zeros(T + 1); dy = np.zeros(T + 1)
    for t in range(1, T + 1):
        da = lambda k: (a[t - k] - a[t - k - 1]) if t - k >= 1 else (0.0 if t - k < 0 else a[0] - 0.0)
        e = lam * y[t - 1] + gam * a[t - 1] + sum(phi[i - 1] * (dy[t - i] if t - i >= 1 else 0.0) for i in range(1, p + 1)) + sum(psi[i] * da(i) for i in range(r + 1))
        dy[t] = e; y[t] = y[t - 1] + e
    return 100 * y[1:]            # per cent (log points) at weeks 1..T

rows = []
for ch in CHAINS:
    o, (d, dd, X, m, (p, q, r)) = run(CH[ch], IDX, "prod", "level", False, with_sym=False); names = list(X.columns)
    hac = sm.OLS(dd["dy"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 4, "use_correction": True}); b = hac.params.values; V = hac.cov_params().values
    draws = rng.multivariate_normal(b, V, size=2000)
    for lab, s in shocks.items():
        for h in (1, 4):
            pt = irf(b, names, p, r, s, h); D = np.array([irf(x, names, p, r, s, h) for x in draws])
            for wk in (1, 4, 12, 26):
                lo, hi = np.percentile(D[:, wk - 1], [5, 95]); rows.append(dict(chain=ch, shock=lab, size=s, duration_wks=h, week=wk, response_pct=pt[wk - 1], lo90=lo, hi90=hi, band_incl_zero=(lo <= 0 <= hi)))
    print(ch, f"{time.time()-t0:.0f}s", flush=True)
T = pd.DataFrame(rows); T.to_csv(TAB / "tableA6_dynamic_responses.csv", index=False)
print(T[(T.shock == "99th pct") & (T.week.isin([4, 26]))].round(3).to_string(index=False))
print("\nShare of (chain x shock x duration x week) cells whose 90% band includes zero:", round(T.band_incl_zero.mean(), 3))
print("Max |response| across all cells (per cent):", round(T.response_pct.abs().max(), 2))

# ---------------- A1 long-run coefficients (unrestricted NARDL, delta method with HAC) + A13 equal weights
rows = []
for ch in CHAINS:
    o, (d, dd, X, m, (p, q, r)) = run(CH[ch], IDX, "prod", "nardl", False, with_sym=False); names = list(X.columns)
    hac = sm.OLS(dd["dy"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 4, "use_correction": True}); b = hac.params.values; V = hac.cov_params().values
    il = names.index("y_L1"); out = dict(chain=ch, n=len(dd), ECT=b[il])
    for nm in ("x_L1", "P_L1", "N_L1"):
        j = names.index(nm); th = -b[j] / b[il]; gr = np.zeros(len(b)); gr[j] = -1 / b[il]; gr[il] = b[j] / b[il] ** 2; se = float(np.sqrt(gr @ V @ gr)); out[nm + "_theta"] = th; out[nm + "_se"] = se
    rows.append(out)
pd.DataFrame(rows).to_csv(TAB / "tableA1_longrun_coefficients.csv", index=False); print("\nA1 long-run coefficients (delta method, HAC)"); print(pd.DataFrame(rows).round(3).to_string(index=False))

a = IDX; e = IDX_EQ; j = pd.concat([a["NASCDI"].rename("m"), e["NASCDI"].rename("e")], axis=1).dropna()
dj = j.diff().dropna(); fl = lambda s: (s <= s.min() + 1e-9).mean() * 100
A13 = dict(r_levels=j.corr().iloc[0, 1], r_changes=dj.corr().iloc[0, 1], r_P=np.corrcoef(a["P"], e["P"])[0, 1], r_N=np.corrcoef(a["N"], e["N"])[0, 1], weeks_floor_main=fl(a["NASCDI"]), weeks_floor_eq=fl(e["NASCDI"]))
pd.DataFrame([A13]).to_csv(TAB / "tableA13_equal_weights.csv", index=False); print("\nA13", {k: round(v, 3) for k, v in A13.items()})
print("done", f"{time.time()-t0:.0f}s")
