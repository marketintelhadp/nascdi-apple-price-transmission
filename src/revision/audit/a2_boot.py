import sys, time, importlib.util, itertools, numpy as np, pandas as pd, statsmodels.api as sm
spec = importlib.util.spec_from_file_location("rev_lib", __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "rev_lib.py"))
L = importlib.util.module_from_spec(spec); spec.loader.exec_module(L)
from statsmodels.stats.diagnostic import acorr_breusch_godfrey

def design(f, p, q, r, spec):
    if spec == "xonly":
        d, y, X = L.build_design(f, p, q, 0, "sym")
        X = X.drop(columns=[c for c in X.columns if c.startswith("Z_") or c.startswith("dZ_")])
        d = d.drop(columns=[c for c in d.columns if c.startswith("Z_") or c.startswith("dZ_")])
        return d, y, X
    return L.build_design(f, p, q, r, spec)

def levels(spec):
    return ["y_L1", "x_L1"] + (["P_L1", "N_L1"] if spec == "asym" else (["Z_L1"] if spec == "sym" else []))

def stats_F_t(y, X, spec):
    m = sm.OLS(y, X).fit()
    lev = levels(spec)
    R = np.zeros((len(lev), X.shape[1]))
    for i, n in enumerate(lev): R[i, list(X.columns).index(n)] = 1
    return float(m.f_test(R).fvalue), float(m.tvalues["y_L1"]), m

def select(f, spec, pmax=3, qmax=3, rmax=3):
    rr = range(rmax + 1) if spec in ("asym", "sym") else [0]
    dref, _, _ = design(f, pmax, qmax, rmax if spec != "xonly" else 0, spec)
    ci = dref.index; best = None
    for p, q, r in itertools.product(range(pmax + 1), range(qmax + 1), rr):
        d, y, X = design(f, p, q, r, spec)
        if len(ci.intersection(d.index)) < len(ci): continue
        y, X = y.loc[ci], X.loc[ci]
        if X.shape[1] >= len(y) - 5: continue
        m = sm.OLS(y, X).fit()
        try: bg = float(acorr_breusch_godfrey(m, nlags=4)[1])
        except Exception: bg = np.nan
        if not bg > 0.05: continue
        if best is None or m.aic < best[0]: best = (m.aic, p, q, r)
    if best is None: best = (0, 1, 1, 0 if spec=="xonly" else 1)
    return best[1:]

def boot_null(f, p, q, r, spec, B, rng):
    d, y, X = design(f, p, q, r, spec)
    F0, t0, _ = stats_F_t(y, X, spec)
    lev = levels(spec)
    keep = [c for c in X.columns if c not in lev]
    Xr = X[keep]
    mr = sm.OLS(y, Xr).fit()
    e = mr.resid.values
    # reconstruct path: only dy-lag terms feed back recursively
    idx = X.index
    grid = f.index
    pos = {t: i for i, t in enumerate(grid)}
    yv = f["y"].values.copy()
    dyact = np.r_[np.nan, np.diff(yv)]  # NaN where gap/missing
    ex_cols = [c for c in keep if not c.startswith("dy_L")]
    exog_fixed = (Xr[ex_cols] @ mr.params[ex_cols]).reindex(grid)   # const + dx/dP/dN terms (actual)
    phi = [mr.params.get(f"dy_L{i}", 0.0) for i in range(1, p + 1)]
    insample = pd.Series(False, index=grid); insample.loc[idx] = True
    ev = pd.Series(e, index=idx)
    Fb, tb = [], []
    for _ in range(B):
        v = rng.choice([-1.0, 1.0], size=len(ev))
        es = pd.Series(ev.values * v, index=idx)
        dys = dyact.copy()
        for i, t in enumerate(grid):
            if insample.iloc[i]:
                val = exog_fixed.iloc[i] + es.loc[t]
                for k, ph in enumerate(phi, 1):
                    val += ph * dys[i - k]
                dys[i] = val
        ys = np.full(len(grid), np.nan)
        last = None
        for i in range(len(grid)):
            if np.isnan(yv[i]): continue
            if last is None: ys[i] = yv[i]
            elif i - last == 1 and not np.isnan(dys[i]): ys[i] = ys[last] + dys[i]
            else: ys[i] = ys[last] + (yv[i] - yv[last])   # gap: carry actual jump
            last = i
        fb = f.copy(); fb["y"] = ys
        try:
            d2, y2, X2 = design(fb, p, q, r, spec)
            Fs, ts, _ = stats_F_t(y2, X2, spec)
            Fb.append(Fs); tb.append(ts)
        except Exception: pass
    Fb, tb = np.array(Fb), np.array(tb)
    return F0, t0, (1 + (Fb >= F0).sum()) / (1 + len(Fb)), (1 + (tb <= t0).sum()) / (1 + len(tb)), np.quantile(Fb, .95), np.quantile(tb, .05)

if __name__ == "__main__":
    B = int(sys.argv[1]); chain_sel = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "-" else None; REV = len(sys.argv) > 3 and sys.argv[3] == "rev"
    nas = L.load_nascdi(); rows = []
    for name, path in L.chains():
        if chain_sel and chain_sel not in name: continue
        f = L.chain_frame(path, nas["NASCDI"])
        if REV: f = f.copy(); f["y"], f["x"] = f["x"].copy(), f["y"].copy()
        for spec in ("asym", "sym", "xonly"):
            t0 = time.time()
            p, q, r = select(f, spec)
            rng = np.random.default_rng(3)
            F0, t0_, pF, pt, F95, t05 = boot_null(f, p, q, r, spec, B, rng)
            k = len(levels(spec)) - 1
            d, y, X = design(f, p, q, r, spec)
            cv = L.sim_bounds_cv(len(y), p, q, r if spec != "xonly" else 0, k, reps=2000, seed=11)
            dF = "coint" if F0 > cv["U"]["F05"] else ("none" if F0 < cv["L"]["F05"] else "inc")
            dt = "coint" if t0_ < cv["U"]["t05"] else ("none" if t0_ > cv["L"]["t05"] else "inc")
            rows.append(dict(chain=name, spec=spec, p=p, q=q, r=r, n=len(y), F=F0, t=t0_, simF_L=cv["L"]["F05"], simF_U=cv["U"]["F05"],
                             simt_L=cv["L"]["t05"], simt_U=cv["U"]["t05"], decF=dF, dect=dt, bootF95=F95, bootT05=t05, pF_boot=pF, pt_boot=pt))
            print(rows[-1]["chain"], spec, (p, q, r), "n", len(y), "F %.2f t %.2f | sim U %.2f/%.2f %s %s | boot crit F %.2f t %.2f | p_boot F %.3f t %.3f | %.0fs" %
                  (F0, t0_, cv["U"]["F05"], cv["U"]["t05"], dF, dt, F95, t05, pF, pt, time.time() - t0), flush=True)
    import os; pd.DataFrame(rows).to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "results_revision", "audit", "bounds_bootstrap_" + ("reverse" if REV else "producer") + ".csv"), index=False)
