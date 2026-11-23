"""Appendix A4 and A9 with the DEFINITIONS OF THE ORIGINAL APPENDIX:
A4: sup-Wald (HC3) for a break in the coefficients on the four lagged-level terms, trimming 20%, restricted wild bootstrap (299);
    Chow/Wald at 4 Mar 2018 (Sopore) with wild-bootstrap p; variance ratio post/pre; wild-bootstrap p for the lagged-level block (1,999); ECT s.e. plain/HC3/HAC.
A9: bounds F and t with restricted wild bootstrap under no level relationship (all lagged levels zero), regressors held at sample values (999)."""
import sys, importlib, time
import numpy as np, pandas as pd, statsmodels.api as sm
modname = sys.argv[1]; tag = sys.argv[2]; part = sys.argv[3]
G = importlib.import_module(modname)
from scipy import stats
CH, CHAINS, IDX, run, rng, hc3_wald, cfit, wild_p, lagged_level_test = G.CH, G.CHAINS, G.IDX, G.run, G.rng, G.hc3_wald, G.cfit, G.wild_p, G.lagged_level_test
OUT = G.TAB.parent / f"tables_{tag}"; OUT.mkdir(exist_ok=True); t0 = time.time()

def boot_FT(y, X, names, lv, B=999):
    Rm = np.zeros((len(lv), len(names)))
    for i, nm in enumerate(lv): Rm[i, names.index(nm)] = 1
    j = names.index("y_L1"); k = X.shape[1]; n = len(y); XtXi = np.linalg.inv(X.T @ X)
    def stats_(yy):
        b = XtXi @ X.T @ yy; e = yy - X @ b; s2 = e @ e / (n - k); V = s2 * XtXi; r = Rm @ b
        return float(r @ np.linalg.solve(Rm @ V @ Rm.T, r)) / len(lv), b[j] / np.sqrt(V[j, j])
    Fo, to = stats_(y); b0 = cfit(y, X, Rm); f0 = X @ b0; e0 = y - f0; cF = ct = 0
    for _ in range(B):
        Fb, tb = stats_(f0 + e0 * rng.choice([-1.0, 1.0], size=n)); cF += Fb >= Fo; ct += tb <= to
    return (cF + 1) / (B + 1), (ct + 1) / (B + 1)

if part == "a9":
    LV = {"nardl": ["y_L1", "x_L1", "P_L1", "N_L1"], "level": ["y_L1", "x_L1", "a_L1"], "prices": ["y_L1", "x_L1"]}; rows = []
    for dep in ("prod", "term"):
        for ch in CHAINS:
            for spec in ("nardl", "level", "prices"):
                o, (d, dd, X, m, (p, q, r)) = run(CH[ch], IDX, dep, spec, False, with_sym=False); pF, pt = boot_FT(dd["dy"].values, X.values, list(X.columns), LV[spec])
                bd = lambda c: {"Yes": "coint.", "Mixed": "mixed", "Inconclusive": "inconcl.", "No": "none"}[c]
                fb = "coint." if o["F"] > o["F_hi"] else ("inconcl." if o["F"] > o["F_lo"] else "none"); tb = "coint." if o["t"] < o["t_hi"] else ("inconcl." if o["t"] < o["t_lo"] else "none")
                rows.append(dict(dep=dep, chain=ch, spec=spec, lags=f"{p},{q},{r}" if spec == "nardl" else (f"{p},{q},{r}" if spec == "level" else f"{p},{q}"), n=o["n"], F=o["F"], t=o["t"], F_lo=o["F_lo"], F_hi=o["F_hi"], t_lo=o["t_lo"], t_hi=o["t_hi"], bounds=f"{fb} / {tb}", cat=o["cat"], pF=pF, pt=pt))
        print(dep, f"{time.time()-t0:.0f}s", flush=True)
    T = pd.DataFrame(rows); T.to_csv(OUT / "A9.csv", index=False); print(T.round(3).to_string(index=False))

if part == "a4":
    cut = pd.Timestamp("2018-03-04"); rows = []
    for ch in CHAINS:
        o, (d, dd, X, m, (p, q, r)) = run(CH[ch], IDX, "prod", "nardl", False, with_sym=False); y = dd["dy"].values; Xv = X.values; names = list(X.columns); n, k = Xv.shape
        lvi = [names.index(c) for c in ("y_L1", "x_L1", "P_L1", "N_L1")]; lo, hi = int(0.20 * n), int(0.80 * n)
        def wald_at(yy, t):
            Z = Xv[:, lvi] * (np.arange(n) >= t)[:, None]; XZ = np.c_[Xv, Z]
            try: Rm = np.zeros((len(lvi), XZ.shape[1])); Rm[np.arange(len(lvi)), k + np.arange(len(lvi))] = 1; return hc3_wald(yy, XZ, Rm)
            except np.linalg.LinAlgError: return 0.0
        def supw(yy): vals = [wald_at(yy, t) for t in range(lo, hi)]; i = int(np.argmax(vals)); return vals[i], lo + i
        so, ti = supw(y); b0 = np.linalg.lstsq(Xv, y, rcond=None)[0]; f0 = Xv @ b0; e0 = y - f0; B = 299; c = 0
        for _ in range(B): c += supw(f0 + e0 * rng.choice([-1.0, 1.0], size=n))[0] >= so
        dates = dd.index; sup_date = dates[ti].date(); tk = int(np.searchsorted(dates.values, np.datetime64(cut))); chow = chp = np.nan; npre = tk; npost = n - tk
        if 10 <= tk <= n - 10:
            chow = wald_at(y, tk); chp = (sum(wald_at(f0 + e0 * rng.choice([-1.0, 1.0], size=n), tk) >= chow for _ in range(B)) + 1) / (B + 1)
        e = m.resid.values; pre, post = e[:tk], e[tk:]
        vr = vrp = np.nan
        if len(pre) >= 10 and len(post) >= 10: vr = post.var(ddof=1) / pre.var(ddof=1); vrp = 2 * min(stats.f.cdf(vr, len(post) - 1, len(pre) - 1), stats.f.sf(vr, len(post) - 1, len(pre) - 1))
        F, Rl = lagged_level_test(m, X, "nardl"); pblock = wild_p(y, Xv, Rl)
        hc3 = sm.OLS(dd["dy"], X).fit(cov_type="HC3"); hac = sm.OLS(dd["dy"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 4, "use_correction": True})
        rows.append(dict(chain=ch, n=n, supW=so, supW_date=str(sup_date), supW_p=(c + 1) / (B + 1), chow_W=chow, chow_p=chp, n_pre=npre, n_post=npost, var_ratio=vr, var_p=vrp, p_block=pblock, se_plain=m.bse["y_L1"], se_HC3=hc3.bse["y_L1"], se_HAC=hac.bse["y_L1"]))
        print(ch, f"{time.time()-t0:.0f}s", flush=True)
    T = pd.DataFrame(rows); T.to_csv(OUT / "A4.csv", index=False); print(T.round(3).to_string(index=False))
