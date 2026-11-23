"""Remaining index-dependent tables on the rebuilt index. Usage: python regen_rest.py a9 | a10 | a11 | a4"""
import sys, time
import numpy as np, pandas as pd, statsmodels.api as sm
import regen as G
from regen import *

which = sys.argv[1]; t0 = time.time()
Bw = 1999
def tboot(d, dd, X, spec, B=Bw):
    """restricted wild bootstrap p for the t-test on y_L1 (H0: lambda=0), regressors held at sample values, one-sided (lower)."""
    names = list(X.columns); Xv = X.values; y = dd["dy"].values
    Rm = np.zeros((1, len(names))); Rm[0, names.index("y_L1")] = 1
    b0 = cfit(y, Xv, Rm); f0 = Xv @ b0; e0 = y - f0; XtXi = np.linalg.inv(Xv.T @ Xv); j = names.index("y_L1")
    def tstat(yy):
        b = XtXi @ Xv.T @ yy; e = yy - Xv @ b; s2 = e @ e / (len(yy) - Xv.shape[1]); return b[j] / np.sqrt(s2 * XtXi[j, j])
    tobs = tstat(y); cnt = sum(tstat(f0 + e0 * rng.choice([-1.0, 1.0], size=len(y))) <= tobs for _ in range(B))
    return (cnt + 1) / (B + 1)

if which == "a9":
    rows = []
    for ch in CHAINS:
        for dep in ("prod", "term"):
            for spec in ("nardl", "level", "prices"):
                o, (d, dd, X, m, _) = run(CH[ch], IDX, dep, spec, False, with_sym=False)
                rows.append(dict(chain=ch, dep=dep, spec=spec, **{k: o[k] for k in ("n", "p", "q", "r", "ECT", "F", "F_lo", "F_hi", "t", "t_lo", "t_hi", "cat")}, t_pboot=tboot(d, dd, X, spec)))
        print(ch, f"{time.time()-t0:.0f}s", flush=True)
    T = pd.DataFrame(rows); T.to_csv(TAB / "tableA9_bounds_both_directions.csv", index=False)
    print(T.round(3).to_string(index=False))
    both = T[T.cat == "Yes"]; print("\nChains with both statistics beyond the bounds:\n", both[["chain", "dep", "spec", "F", "F_hi", "t", "t_hi", "t_pboot"]].round(3).to_string(index=False))

if which == "a10":
    def filt(df, k):
        df = df.copy()
        for c in ("log_price_prod", "log_price_term"):
            v = df[c].values; run_id = np.r_[0, np.cumsum(np.abs(np.diff(v)) > 1e-12)]; sz = pd.Series(run_id).map(pd.Series(run_id).value_counts()).values
            first = np.r_[True, np.diff(run_id) != 0]; drop = (sz >= k) & ~first; df.loc[drop, "log_price_prod" if c == "log_price_prod" else "log_price_term"] = np.nan
        return df
    rows = []
    for k in (6, 10):
        for ch in CHAINS:
            df = filt(CH[ch], k); o, _ = run(df, IDX, "prod", "nardl", False, with_sym=False)
            rows.append(dict(minrun=k, chain=ch, n_obs_kept=int(df[["log_price_prod", "log_price_term"]].notna().all(axis=1).sum()), **{c: o[c] for c in ("n", "p", "q", "r", "ECT", "F", "F_lo", "F_hi", "t", "t_lo", "t_hi", "cat")}))
        print("minrun", k, f"{time.time()-t0:.0f}s", flush=True)
    T = pd.DataFrame(rows); T.to_csv(TAB / "tableA10_repeated_run_filter.csv", index=False); print(T.round(3).to_string(index=False))

if which == "a11":
    sd = IDX["NASCDI"].diff().std(); rows = []; store = {}
    for ch in CHAINS:
        o, (d, dd, X, m, (p, q, r)) = run(CH[ch], IDX, "prod", "nardl", False, with_sym=False); names = list(X.columns); y = dd["dy"].values; Xv = X.values
        sr = [names.index(f"dP_L{i}") for i in range(r + 1)] + [names.index(f"dN_L{i}") for i in range(r + 1)]; Rs = np.zeros((len(sr), len(names)))
        for a, jx in enumerate(sr): Rs[a, jx] = 1
        Rl = np.zeros((2, len(names))); Rl[0, names.index("P_L1")] = 1; Rl[1, names.index("N_L1")] = 1
        Rj = np.vstack([Rs, Rl]); hac = sm.OLS(dd["dy"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 4, "use_correction": True})
        out = dict(chain=ch, n=len(dd))
        for lab, Rm in (("SR_zero", Rs), ("LR_zero", Rl), ("Joint_zero", Rj)):
            w, ph = hac_wald(hac, Rm); out[lab + "_pHAC"] = ph; out[lab + "_pboot"] = wild_p(y, Xv, Rm)
        # detectable effect: level spec, impact + cumulated short-run response of dlog(price) to a one-SD weekly index change, 80% power at 5% (2.8 SE)
        o2, (d2, dd2, X2, m2, (p2, q2, r2)) = run(CH[ch], IDX, "prod", "level", False, with_sym=False); n2 = list(X2.columns)
        v = np.zeros(len(n2));
        for i in range(r2 + 1): v[n2.index(f"da_L{i}")] = 1
        h2 = sm.OLS(dd2["dy"], X2).fit(cov_type="HAC", cov_kwds={"maxlags": 4, "use_correction": True}); est = float(v @ h2.params.values); se = float(np.sqrt(v @ h2.cov_params().values @ v))
        out.update(sr_sum_coef=est, sr_sum_se=se, effect_per_SD_pct=100 * est * sd, MDE80_per_SD_pct=100 * 2.8 * se * sd)
        rows.append(out); store[ch] = (dd["dy"].index, y, Xv, Rj, dd.index); print(ch, f"{time.time()-t0:.0f}s", flush=True)
    T = pd.DataFrame(rows); T.to_csv(TAB / "tableA11_index_inclusion.csv", index=False); print(T.round(3).to_string(index=False)); print("SD weekly NASCDI change", round(sd, 3))
    # pooled test: sum of HC3 Wald statistics (joint zero) across chains, restricted wild bootstrap with a common week-level multiplier
    def stat_all(ys):
        return sum(hc3_wald(ys[ch], store[ch][2], store[ch][3]) for ch in CHAINS)
    f0 = {}; e0 = {}
    for ch in CHAINS:
        _, y, Xv, Rj, ix = store[ch]; b0 = cfit(y, Xv, Rj); f0[ch] = Xv @ b0; e0[ch] = y - f0[ch]
    obs = stat_all({ch: store[ch][1] for ch in CHAINS}); weeks = sorted(set().union(*[set(store[ch][4]) for ch in CHAINS])); cnt = 0; B = 999
    for _ in range(B):
        mult = pd.Series(rng.choice([-1.0, 1.0], size=len(weeks)), index=weeks)
        cnt += stat_all({ch: f0[ch] + e0[ch] * mult.reindex(store[ch][4]).values for ch in CHAINS}) >= obs
    print(f"\nPooled sum of eight HC3 Wald statistics (joint zero) = {obs:.2f}; common-multiplier wild-bootstrap p = {(cnt+1)/(B+1):.3f}")
    pd.DataFrame([dict(pooled_wald_sum=obs, pboot=(cnt + 1) / (B + 1), B=B)]).to_csv(TAB / "tableA11_pooled.csv", index=False)

if which == "a4":
    cut = pd.Timestamp("2018-03-04"); rows = []
    for ch in CHAINS:
        o, (d, dd, X, m, (p, q, r)) = run(CH[ch], IDX, "prod", "level", False, with_sym=False); y = dd["dy"].values; Xv = X.values; n = len(y); k = Xv.shape[1]; names = list(X.columns)
        sl = [names.index(c) for c in ("y_L1", "x_L1", "a_L1", "dx_L0") + tuple(f"da_L{i}" for i in range(r + 1)) if c in names]; sl = sorted(set(sl))
        def chowF(yy, t):
            Z = Xv[:, sl] * (np.arange(n) >= t)[:, None]; XZ = np.c_[Xv, Z]
            b1 = np.linalg.lstsq(XZ, yy, rcond=None)[0]; s1 = ((yy - XZ @ b1) ** 2).sum(); b0 = np.linalg.lstsq(Xv, yy, rcond=None)[0]; s0 = ((yy - Xv @ b0) ** 2).sum()
            return ((s0 - s1) / len(sl)) / (s1 / (n - XZ.shape[1]))
        lo, hi = int(0.15 * n), int(0.85 * n); grid = range(lo, hi)
        supF = lambda yy: max(chowF(yy, t) for t in grid); sobs = supF(y)
        b0 = np.linalg.lstsq(Xv, y, rcond=None)[0]; f0 = Xv @ b0; e0 = y - f0; B = 399
        cnt = sum(supF(f0 + e0 * rng.choice([-1.0, 1.0], size=n)) >= sobs for _ in range(B))
        t_known = int(np.searchsorted(dd.index.values, np.datetime64(cut))); kn = np.nan; knp = np.nan
        if lo <= t_known <= hi:
            kn = chowF(y, t_known); knp = (sum(chowF(f0 + e0 * rng.choice([-1.0, 1.0], size=n), t_known) >= kn for _ in range(B)) + 1) / (B + 1)
        e = m.resid.values; pre, post = e[: max(t_known, 1)], e[max(t_known, 1):]
        vr = (post.var(ddof=1) / pre.var(ddof=1)) if len(pre) > 5 and len(post) > 5 else np.nan; vrp = 2 * min(stats.f.cdf(vr, len(post) - 1, len(pre) - 1), stats.f.sf(vr, len(post) - 1, len(pre) - 1)) if vr == vr else np.nan
        mb = run(CH[ch], IDX, "prod", "nardl", False, with_sym=False)[1]; X2 = mb[2]; m2 = mb[3]; hc3 = sm.OLS(mb[1]["dy"], X2).fit(cov_type="HC3"); hac = sm.OLS(mb[1]["dy"], X2).fit(cov_type="HAC", cov_kwds={"maxlags": 4, "use_correction": True})
        rows.append(dict(chain=ch, n=n, supF=sobs, supF_pboot=(cnt + 1) / (B + 1), known_F=kn, known_pboot=knp, n_pre=len(pre), n_post=len(post), var_ratio=vr, var_ratio_p=vrp, ECT=m2.params["y_L1"], se_plain=m2.bse["y_L1"], se_HC3=hc3.bse["y_L1"], se_HAC=hac.bse["y_L1"]))
        print(ch, f"{time.time()-t0:.0f}s", flush=True)
    T = pd.DataFrame(rows); T.to_csv(TAB / "tableA4_stability.csv", index=False); print(T.round(3).to_string(index=False))
print("done", f"{time.time()-t0:.0f}s")
