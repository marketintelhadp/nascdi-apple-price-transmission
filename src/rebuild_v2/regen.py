"""
Regenerate the index-dependent tables of the revised manuscript on the REBUILT index (data/nascdi_news_v2).
Conventions follow the revised manuscript: NASCDI+ = cumulated rises (>=0); NASCDI- = cumulated declines, NON-POSITIVE (Shin et al. 2014),
so long-run symmetry means equal coefficients on the lagged partial sums and short-run symmetry equal sums of the short-run coefficients.
Calendar-week (strict) design; lags (p, q, r) chosen per equation by AIC (0-3) on a common sample subject to Breusch-Godfrey(4) not rejecting at 5%.
Bounds: finite-sample critical values simulated per equation (I(0)/I(1) bounds); HAC = Newey-West, 4 lags, small-sample correction.
"""
import sys, glob, re, itertools, warnings, time
from pathlib import Path
import numpy as np, pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import acorr_breusch_godfrey, acorr_ljungbox, het_breuschpagan, het_white, het_arch, linear_reset, breaks_cusumolsresid
from statsmodels.stats.stattools import jarque_bera
from statsmodels.stats.outliers_influence import variance_inflation_factor
sys.stdout.reconfigure(encoding="utf-8"); warnings.filterwarnings("ignore")
import os
REPO = Path(os.environ.get("NASCDI_ROOT", Path(__file__).resolve().parents[2]))
TAB = REPO / "results_rebuild_v2" / "tables"; TAB.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(20261120)
R_SIM, B_BOOT = 2000, 1999
LMAX = 3

# ------------------------------------------------------------------ data
def load_index(path):
    w = pd.read_csv(path, parse_dates=["week_end"]).sort_values("week_end").drop_duplicates("week_end").set_index("week_end")
    d = w["NASCDI"].diff()
    return pd.DataFrame({"NASCDI": w["NASCDI"], "P": np.where(d > 0, d, 0.0).cumsum(), "N": -np.where(d < 0, -d, 0.0).cumsum()}, index=w.index)   # N non-positive
IDX = load_index(REPO / "data/nascdi_news_v2/nascdi_news_weekly.csv")
IDX_EQ = load_index(REPO / "data/nascdi_news_v2/nascdi_news_weekly_equalweights.csv")
def parse(p):
    m = re.match(r"weekly_model_(.+)_(American|Delicious)_(A|B)_posneg", Path(p).stem); return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
CH = {parse(f): pd.read_csv(f, parse_dates=["week_end"]).sort_values("week_end") for f in sorted(glob.glob(str(REPO / "data/model_news_v2/weekly_model_*_posneg.csv")))}
CHAINS = list(CH)

# ------------------------------------------------------------------ design
def build(df, idx, dep="prod", spec="nardl", covid=False):
    s = df.set_index("week_end"); grid = pd.date_range(s.index.min(), s.index.max(), freq="W-SUN")
    yv, xv = ("log_price_prod", "log_price_term") if dep == "prod" else ("log_price_term", "log_price_prod")
    g = pd.DataFrame(index=grid); g["y"] = s[yv].reindex(grid); g["x"] = s[xv].reindex(grid)
    g["P"] = idx["P"].reindex(grid); g["N"] = idx["N"].reindex(grid); g["a"] = g["P"] + g["N"]     # a = NASCDI_t - NASCDI_0
    d = pd.DataFrame(index=grid)
    for k in ("y", "x", "P", "N", "a"): d["d" + k] = g[k].diff()
    d["y_L1"], d["x_L1"], d["P_L1"], d["N_L1"], d["a_L1"] = (g[k].shift(1) for k in ("y", "x", "P", "N", "a"))
    for l in range(0, LMAX + 1):
        for k in ("y", "x", "P", "N", "a"): d[f"d{k}_L{l}"] = d["d" + k].shift(l)
    if covid: d["D"] = ((grid >= "2020-03-01") & (grid <= "2021-12-31")).astype(float)
    return d

def cols(spec, p, q, r, covid=False):
    base = {"nardl": ["y_L1", "x_L1", "P_L1", "N_L1"], "level": ["y_L1", "x_L1", "a_L1"], "prices": ["y_L1", "x_L1"], "sr_sym": ["y_L1", "x_L1", "P_L1", "N_L1"], "lr_sym": ["y_L1", "x_L1", "a_L1"]}[spec]
    c = base + [f"dy_L{i}" for i in range(1, p + 1)] + [f"dx_L{i}" for i in range(q + 1)]
    if spec in ("nardl", "lr_sym"): c += [f"dP_L{i}" for i in range(r + 1)] + [f"dN_L{i}" for i in range(r + 1)]
    if spec in ("level", "sr_sym"): c += [f"da_L{i}" for i in range(r + 1)]
    return c + (["D"] if covid else [])

def fit(d, spec, p, q, r, covid=False):
    c = cols(spec, p, q, r, covid); dd = d.dropna(subset=["dy"] + c); X = sm.add_constant(dd[c], has_constant="add")
    return dd, X, sm.OLS(dd["dy"], X).fit()

def select(d, spec, covid=False):
    """AIC over (p,q,r) in 0..3 on a common sample, subject to BG(4) p >= 0.05; falls back to the best-BG candidate."""
    rr = range(LMAX + 1) if spec != "prices" else [0]
    common = d.dropna(subset=["dy"] + cols(spec, LMAX, LMAX, max(rr), covid)); rows = []
    for p, q, r in itertools.product(range(LMAX + 1), range(LMAX + 1), rr):
        c = cols(spec, p, q, r, covid); k = len(c) + 1
        if len(common) < k + 12: continue
        m = sm.OLS(common["dy"], sm.add_constant(common[c], has_constant="add")).fit(); rows.append((m.aic, p, q, r))
    rows.sort(); best_fail = None
    for aic, p, q, r in rows:
        dd, X, m = fit(d, spec, p, q, r, covid)
        try: bgp = acorr_breusch_godfrey(m, nlags=4)[1]
        except Exception: bgp = np.nan
        if bgp >= 0.05: return (p, q, r), bgp
        if best_fail is None or bgp > best_fail[1]: best_fail = ((p, q, r), bgp)
    return best_fail

# ------------------------------------------------------------------ simulated bounds (k long-run regressors)
_cache = {}
def sim_bounds(n, p, qs, k, R=R_SIM):
    key = (n, p, tuple(qs), k)
    if key in _cache: return _cache[key]
    g = np.random.default_rng(20261120 + 1000 * n + 100 * p + 10 * sum(qs) + k)
    ncols = 1 + 1 + k + p + sum(q + 1 for q in qs); mx = max([p] + list(qs)); N = n + mx + 1; res = {}
    for bound in ("I0", "I1"):
        Fs, Ts = [], []
        for chunk in range(0, R, 400):
            B = min(400, R - chunk); dy = g.standard_normal((B, N)); y = np.cumsum(dy, axis=1)
            if bound == "I1": dxs = g.standard_normal((B, k, N)); xs = np.cumsum(dxs, axis=2)
            else: xs = g.standard_normal((B, k, N)); dxs = np.diff(xs, axis=2, prepend=xs[:, :, :1])
            rows = np.arange(mx + 1, N)[-n:]
            cl = [np.ones((B, n)), y[:, rows - 1]] + [xs[:, j, rows - 1] for j in range(k)]
            for i in range(1, p + 1): cl.append(dy[:, rows - i])
            for j in range(k):
                for i in range(qs[j] + 1): cl.append(dxs[:, j, rows - i])
            X = np.stack(cl, axis=2); Y = dy[:, rows]
            XtX = np.einsum("bnk,bnl->bkl", X, X); Xty = np.einsum("bnk,bn->bk", X, Y)
            inv = np.linalg.inv(XtX); beta = np.einsum("bkl,bl->bk", inv, Xty)
            e = Y - np.einsum("bnk,bk->bn", X, beta); s2 = (e ** 2).sum(1) / (n - ncols); cov = inv * s2[:, None, None]
            bb = beta[:, 1:2 + k]; V = cov[:, 1:2 + k, 1:2 + k]; W = np.einsum("bi,bij,bj->b", bb, np.linalg.inv(V), bb)
            Fs.append(W / (k + 1)); Ts.append(beta[:, 1] / np.sqrt(cov[:, 1, 1]))
        res[bound] = (np.concatenate(Fs), np.concatenate(Ts))
    out = dict(F0=np.percentile(res["I0"][0], 95), F1=np.percentile(res["I1"][0], 95), t0=np.percentile(res["I0"][1], 5), t1=np.percentile(res["I1"][1], 5))
    _cache[key] = out; return out
def category(F, t, c):
    fb = "yes" if F > c["F1"] else ("mid" if F > c["F0"] else "no"); tb = "yes" if t < c["t1"] else ("mid" if t < c["t0"] else "no")
    if fb == "yes" and tb == "yes": return "Yes"
    if fb == "yes" or tb == "yes": return "Mixed"
    if fb == "mid" or tb == "mid": return "Inconclusive"
    return "No"

# ------------------------------------------------------------------ inference helpers
def lagged_level_test(m, X, spec):
    names = list(X.columns); lv = {"nardl": ["y_L1", "x_L1", "P_L1", "N_L1"], "level": ["y_L1", "x_L1", "a_L1"], "prices": ["y_L1", "x_L1"], "sr_sym": ["y_L1", "x_L1", "P_L1", "N_L1"], "lr_sym": ["y_L1", "x_L1", "a_L1"]}[spec]
    Rm = np.zeros((len(lv), len(names)))
    for i, nm in enumerate(lv): Rm[i, names.index(nm)] = 1
    return float(np.asarray(m.f_test(Rm).fvalue)), Rm
def hc3_wald(y, X, Rm):
    b = np.linalg.lstsq(X, y, rcond=None)[0]; e = y - X @ b; XtXi = np.linalg.inv(X.T @ X)
    h = np.einsum("ij,jk,ik->i", X, XtXi, X); u = e / (1 - h); meat = (X * u[:, None]).T @ (X * u[:, None])
    V = XtXi @ meat @ XtXi; r = Rm @ b; return float(r @ np.linalg.inv(Rm @ V @ Rm.T) @ r)
def cfit(y, X, Rm):
    XtXi = np.linalg.inv(X.T @ X); b = XtXi @ X.T @ y
    return b - XtXi @ Rm.T @ np.linalg.inv(Rm @ XtXi @ Rm.T) @ (Rm @ b)
def wild_p(y, X, Rm, B=B_BOOT):
    obs = hc3_wald(y, X, Rm); b0 = cfit(y, X, Rm); f0 = X @ b0; e0 = y - f0; cnt = 0
    for _ in range(B): cnt += hc3_wald(f0 + e0 * rng.choice([-1.0, 1.0], size=len(y)), X, Rm) >= obs
    return (cnt + 1) / (B + 1)
def hac_wald(m_hac, Rm):
    t = m_hac.wald_test(Rm, scalar=True, use_f=False); return float(t.statistic), float(t.pvalue)

def symmetry(d, X, dd, p, q, r):
    names = list(X.columns); y = dd["dy"].values; Xv = X.values
    RL = np.zeros((1, len(names))); RL[0, names.index("P_L1")] = 1; RL[0, names.index("N_L1")] = -1          # equal coefficients
    RS = np.zeros((1, len(names)))
    for i in range(r + 1): RS[0, names.index(f"dP_L{i}")] = 1; RS[0, names.index(f"dN_L{i}")] = -1
    RJ = np.vstack([RL, RS]); hac = sm.OLS(dd["dy"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 4, "use_correction": True}); out = {}
    for lab, Rm in (("LR", RL), ("SR", RS), ("J", RJ)):
        w, ph = hac_wald(hac, Rm); out[lab + "_wald"] = w; out[lab + "_pHAC"] = ph; out[lab + "_pboot"] = wild_p(y, Xv, Rm)
    return out
def final_spec(sym):
    lr, sr = sym["LR_pboot"] < 0.05, sym["SR_pboot"] < 0.05
    return "Symmetric" if not lr and not sr else ("SR-symmetric" if not sr and lr else ("LR-symmetric" if not lr and sr else "Unrestricted"))

# ------------------------------------------------------------------ one chain, one specification
def run(df, idx, dep="prod", spec="nardl", covid=False, with_sym=True):
    d = build(df, idx, dep, spec, covid); (p, q, r), bgp = select(d, spec, covid); dd, X, m = fit(d, spec, p, q, r, covid)
    k = {"nardl": 3, "level": 2, "prices": 1, "sr_sym": 3, "lr_sym": 2}[spec]; qs = {"nardl": (q, r, r), "level": (q, r), "prices": (q,), "sr_sym": (q, r, -1), "lr_sym": (q, 2 * r + 1)}[spec]
    F, _ = lagged_level_test(m, X, spec); t = float(m.tvalues["y_L1"]); c5 = sim_bounds(len(dd), p, qs, k); lam = m.params["y_L1"]
    hac = sm.OLS(dd["dy"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 4, "use_correction": True}); ci = hac.conf_int().loc["y_L1"].values
    out = dict(n=len(dd), p=p, q=q, r=r, ECT=lam, ECT_lo=ci[0], ECT_hi=ci[1], half_life=(np.log(0.5) / np.log(1 + lam) if -1 < lam < 0 else np.nan), F=F, F_lo=c5["F0"], F_hi=c5["F1"],
               t=t, t_lo=c5["t0"], t_hi=c5["t1"], cat=category(F, t, c5), R2=m.rsquared, BG4_p=bgp)
    if with_sym and spec == "nardl":
        sym = symmetry(d, X, dd, p, q, r); out.update(sym); out["final"] = final_spec(sym)
    return out, (d, dd, X, m, (p, q, r))

if __name__ == "__main__":
    t0 = time.time()
    # ---------------- Table 6 / 8 / 9 / 7 base (producer dependent, NARDL, unrestricted)
    base = {}; rows6 = []
    for ch in CHAINS:
        o, obj = run(CH[ch], IDX); base[ch] = (o, obj); rows6.append(dict(chain=ch, **o)); print(ch, "done", f"{time.time()-t0:.0f}s", flush=True)
    T6 = pd.DataFrame(rows6); T6.to_csv(TAB / "table6_8_baseline.csv", index=False)
    print("\nTable 6 / 8 (rebuilt index)"); print(T6[["chain", "n", "p", "q", "r", "ECT", "half_life", "F", "F_lo", "F_hi", "t", "t_lo", "t_hi", "cat", "R2"]].round(3).to_string(index=False))
    print(T6[["chain", "LR_wald", "LR_pHAC", "LR_pboot", "SR_wald", "SR_pHAC", "SR_pboot", "J_pHAC", "J_pboot", "final"]].round(3).to_string(index=False))

    # ---------------- Table 9 diagnostics
    rows9 = []
    for ch in CHAINS:
        d, dd, X, m, (p, q, r) = base[ch][1]; e = m.resid
        def safe(f):
            try: return float(f())
            except Exception: return np.nan
        n, kx = len(dd), X.shape[1]
        w = []  # recursive residuals for CUSUM of squares
        Xv, yv = X.values, dd["dy"].values
        for i in range(kx, n):
            b = np.linalg.lstsq(Xv[:i], yv[:i], rcond=None)[0]; xi = Xv[i]; A = np.linalg.pinv(Xv[:i].T @ Xv[:i])
            w.append((yv[i] - xi @ b) / np.sqrt(1 + xi @ A @ xi))
        w = np.array(w); S = np.cumsum(w ** 2) / (w ** 2).sum(); exp = np.arange(1, len(w) + 1) / len(w); dev = float(np.abs(S - exp).max()); band = 1.358 / np.sqrt(len(w) / 2)
        cb = breaks_cusumolsresid(np.asarray(e), ddof=kx)
        rows9.append(dict(chain=ch, LB8_p=safe(lambda: acorr_ljungbox(e, lags=[8], return_df=True)["lb_pvalue"].iloc[0]), BG4_p=safe(lambda: acorr_breusch_godfrey(m, nlags=4)[1]),
                          BP_p=safe(lambda: het_breuschpagan(e, m.model.exog)[1]), White_p=safe(lambda: het_white(e, m.model.exog)[1]), ARCH4_p=safe(lambda: het_arch(e, nlags=4)[1]),
                          RESET_p=safe(lambda: linear_reset(m, power=[2, 3], use_f=True).pvalue), JB_p=safe(lambda: jarque_bera(e)[1]),
                          CUSUM_p=float(cb[1]), CUSUMSQ_dev=dev, CUSUMSQ_band=band, CUSUMSQ_breach=("Yes" if dev > band else "No")))
    T9 = pd.DataFrame(rows9); T9.to_csv(TAB / "table9_diagnostics.csv", index=False); print("\nTable 9"); print(T9.round(3).to_string(index=False))

    # ---------------- Table 7 identification
    rows7 = []
    for ch in CHAINS:
        d, dd, X, m, _ = base[ch][1]; Xi = dd[["y_L1", "x_L1", "P_L1", "N_L1"]]
        Xc = sm.add_constant(Xi, has_constant="add").values
        vifP = variance_inflation_factor(Xc, 3); vifN = variance_inflation_factor(Xc, 4); cond = float(np.linalg.cond(Xc))
        tt = np.arange(len(dd)); r2t = sm.OLS(dd["P_L1"].values, sm.add_constant(tt)).fit().rsquared
        Xr = sm.add_constant(dd[["y_L1", "x_L1", "a_L1", "P_L1"]], has_constant="add").values; vifr = variance_inflation_factor(Xr, 4)
        rows7.append(dict(chain=ch, corr_PN=float(np.corrcoef(dd["P_L1"], dd["N_L1"])[0, 1]), VIF_P=vifP, VIF_N=vifN, cond=cond, R2_trend_P=r2t, VIF_P_reparam=vifr))
    T7 = pd.DataFrame(rows7); T7.to_csv(TAB / "table7_identification.csv", index=False); print("\nTable 7"); print(T7.round(3).to_string(index=False))

    # ---------------- Table 3 / 4 / A8 descriptive
    allw = pd.concat([CH[c][["week_end"]].assign(chain=c) for c in CHAINS]); mw = IDX.reindex(allw["week_end"])
    print("\nTable 3 (NASCDI block, over the %d matched observations):" % len(allw));
    print(f"  NASCDI mean {mw.NASCDI.mean():.2f} SD {mw.NASCDI.std():.2f} range {mw.NASCDI.min():.2f}-{mw.NASCDI.max():.2f}; NASCDI+ mean/SD/max {mw.P.mean():.2f}/{mw.P.std():.2f}/{mw.P.max():.2f}; NASCDI- mean/SD/min {mw.N.mean():.2f}/{mw.N.std():.2f}/{mw.N.min():.2f}")
    uw = IDX.reindex(pd.Index(sorted(set(allw.week_end)))); print(f"  distinct matched weeks: {len(uw)}; NASCDI mean {uw.NASCDI.mean():.2f} SD {uw.NASCDI.std():.2f}")
    dch = IDX["NASCDI"].diff().dropna(); pos, neg = dch[dch > 0], -dch[dch < 0]
    A8 = pd.DataFrame({"stat": ["count", "mean", "std", "min", "25%", "50%", "75%", "max"], "all": dch.describe().values, "before 2018-03-04": dch[dch.index < "2018-03-04"].describe().values, "from 2018-03-04": dch[dch.index >= "2018-03-04"].describe().values})
    A8.to_csv(TAB / "tableA8a_weekly_change.csv", index=False); print(A8.round(2).to_string(index=False))
    pct = pd.DataFrame({"percentile": [50, 75, 90, 95, 99], "increase": [np.percentile(pos, q) for q in (50, 75, 90, 95, 99)], "decrease": [-np.percentile(neg, q) for q in (50, 75, 90, 95, 99)]}); pct.to_csv(TAB / "tableA8b_percentiles.csv", index=False); print(pct.round(2).to_string(index=False))
    pd.DataFrame({"series": ["NASCDI+", "NASCDI-"], "mean": [IDX.P.mean(), IDX.N.mean()], "sd": [IDX.P.std(), IDX.N.std()], "min": [IDX.P.min(), IDX.N.min()], "max": [IDX.P.max(), IDX.N.max()]}).to_csv(TAB / "tableA8c_partial_sums.csv", index=False)
    print("\nrunning robustness variants ...", flush=True)

    # ---------------- Table 10 / A5 robustness variants
    variants = [("baseline", dict(idx=IDX, dep="prod", spec="nardl", covid=False)), ("level (k=2)", dict(idx=IDX, dep="prod", spec="level", covid=False)), ("prices only (k=1)", dict(idx=IDX, dep="prod", spec="prices", covid=False)),
                ("terminal dependent", dict(idx=IDX, dep="term", spec="nardl", covid=False)), ("COVID-19 dummy", dict(idx=IDX, dep="prod", spec="nardl", covid=True)), ("equal-weights index", dict(idx=IDX_EQ, dep="prod", spec="nardl", covid=False))]
    rowsR = []
    for lab, kw in variants:
        for ch in CHAINS:
            if lab == "baseline": o = base[ch][0]
            else: o, _ = run(CH[ch], kw["idx"], kw["dep"], kw["spec"], kw["covid"], with_sym=(kw["spec"] == "nardl"))
            rowsR.append(dict(variant=lab, chain=ch, **o))
        print("  variant done:", lab, f"{time.time()-t0:.0f}s", flush=True)
    TR = pd.DataFrame(rowsR); TR.to_csv(TAB / "table10_A5_robustness.csv", index=False)
    print("\nTable 10 summary");
    for lab, _ in variants:
        s = TR[TR.variant == lab]; Fb = s[s.F > s.F_hi].chain.tolist(); tb = s[s.t < s.t_hi].chain.tolist()
        lr = s[s.get("LR_pboot", pd.Series(1.0, index=s.index)) < 0.05].chain.tolist() if "LR_pboot" in s else []; sr = s[s.get("SR_pboot", pd.Series(1.0, index=s.index)) < 0.05].chain.tolist() if "SR_pboot" in s else []
        print(f"  {lab:22s} ECT {s.ECT.min():.2f} to {s.ECT.max():.2f} | F beyond: {Fb} | t beyond: {tb} | LR sym rejected: {lr} | SR sym rejected: {sr} | categories: {dict(s.cat.value_counts())}")
    print("done", f"{time.time()-t0:.0f}s")
