import sys, importlib.util, itertools, numpy as np, pandas as pd, statsmodels.api as sm
spec = importlib.util.spec_from_file_location("rev_lib", __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "rev_lib.py"))
L = importlib.util.module_from_spec(spec); spec.loader.exec_module(L)
nas = L.load_nascdi()

def exclusion(f, p, q, r, spec, label):
    d, y, X = L.build_design(f, p, q, r, spec)
    names = list(X.columns)
    tgt = [i for i, n in enumerate(names) if n in ("Z_L1", "P_L1", "N_L1") or n.startswith("dZ_L") or n.startswith("dP_L") or n.startswith("dN_L")]
    R = np.zeros((len(tgt), len(names)))
    for k, i in enumerate(tgt): R[k, i] = 1
    W, pb = L.wild_boot_wald(y, X, R, B=1999, seed=9)
    return len(y), len(tgt), W, pb

def shortrun(f, K=4):
    # Delta y on dx(0..2) and dZ(0..K), plus dy lag 1 ; within contiguous weeks
    d = pd.DataFrame(index=f.index)
    d["dy"] = f.y.diff()
    dx = f.x.diff(); dz = f.Z.diff()
    d["dy_L1"] = d.dy.shift(1)
    for j in range(3): d[f"dx_L{j}"] = dx.shift(j)
    for k in range(K + 1): d[f"dZ_L{k}"] = dz.shift(k)
    d = d.dropna()
    X = sm.add_constant(d.drop(columns="dy")); y = d.dy
    names = list(X.columns)
    tgt = [i for i, n in enumerate(names) if n.startswith("dZ_L")]
    R = np.zeros((len(tgt), len(names)))
    for k, i in enumerate(tgt): R[k, i] = 1
    W, pb = L.wild_boot_wald(y, X, R, B=1999, seed=9)
    m = sm.OLS(y, X).fit()
    cum = m.params[[names[i] for i in tgt]].sum()
    return len(y), W, pb, cum

def spread_df(f, p=1, B=1999, seed=2):
    # Delta y = c + a*(y-x)_{t-1} + dx0 + dy lags ; beta=1 imposed ; DF-type bootstrap p for t(a)
    s = f.y - f.x
    d = pd.DataFrame(index=f.index)
    d["dy"] = f.y.diff(); d["s_L1"] = s.shift(1); d["dx0"] = f.x.diff()
    for i in range(1, p + 1): d[f"dy_L{i}"] = d.dy.shift(i)
    d = d.dropna(); X = sm.add_constant(d.drop(columns="dy")); y = d.dy
    m = sm.OLS(y, X).fit(); t0 = m.tvalues["s_L1"]
    # null: alpha = 0, y random walk w/ short-run terms; simulate under restricted fit using wild bootstrap and rebuilt levels
    keep = [c for c in X.columns if c != "s_L1"]
    mr = sm.OLS(y, X[keep]).fit(); e = mr.resid.values; fit = mr.fittedvalues.values
    rng = np.random.default_rng(seed); grid = f.index; yv = f.y.values; dyact = np.r_[np.nan, np.diff(yv)]
    ins = pd.Series(False, index=grid); ins.loc[d.index] = True
    base = (X[[c for c in keep if not c.startswith("dy_L")]] @ mr.params[[c for c in keep if not c.startswith("dy_L")]]).reindex(grid)
    phi = [mr.params.get(f"dy_L{i}", 0.0) for i in range(1, p + 1)]
    ev = pd.Series(e, index=d.index); ts = []
    for _ in range(B):
        es = ev * rng.choice([-1.0, 1.0], size=len(ev)); dys = dyact.copy()
        for i in range(len(grid)):
            if ins.iloc[i]:
                v = base.iloc[i] + es.loc[grid[i]]
                for k, ph in enumerate(phi, 1): v += ph * dys[i - k]
                dys[i] = v
        ys = np.full(len(grid), np.nan); last = None
        for i in range(len(grid)):
            if np.isnan(yv[i]): continue
            if last is None: ys[i] = yv[i]
            elif i - last == 1 and not np.isnan(dys[i]): ys[i] = ys[last] + dys[i]
            else: ys[i] = ys[last] + (yv[i] - yv[last])
            last = i
        fb = f.copy(); fb["y"] = ys
        s2 = fb.y - fb.x
        d2 = pd.DataFrame(index=fb.index); d2["dy"] = fb.y.diff(); d2["s_L1"] = s2.shift(1); d2["dx0"] = fb.x.diff()
        for i in range(1, p + 1): d2[f"dy_L{i}"] = d2.dy.shift(i)
        d2 = d2.dropna(); X2 = sm.add_constant(d2.drop(columns="dy"))
        ts.append(sm.OLS(d2.dy, X2).fit().tvalues["s_L1"])
    ts = np.array(ts)
    return len(y), m.params["s_L1"], t0, (1 + (ts <= t0).sum()) / (1 + len(ts)), np.quantile(ts, .05)

main = pd.read_csv(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "..", "..", "results_revision", "main_base.csv")).set_index("chain")
rows = []
for name, path in L.chains():
    f = L.chain_frame(path, nas["NASCDI"]); p, q, r = int(main.loc[name, "p"]), int(main.loc[name, "q"]), int(main.loc[name, "r"])
    n1, k1, W1, p1 = exclusion(f, p, q, r, "asym", "asym")
    n2, k2, W2, p2 = exclusion(f, p, q, r, "sym", "sym")
    n3, W3, p3, cum = shortrun(f)
    n4, a4, t4, p4, c4 = spread_df(f, p=1)
    rows.append(dict(chain=name, n_asym=n1, p_excl_asym=p1, n_sym=n2, p_excl_sym=p2, n_sr=n3, p_sr_dZ=p3, cum_dZ=cum, n_spread=n4, a_spread=a4, t_spread=t4, p_spread_boot=p4, crit5=c4))
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "..", "..", "results_revision", "audit", "nascdi_exclusion_and_spread.csv"), index=False)
