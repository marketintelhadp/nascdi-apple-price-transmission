import sys, time, json
import numpy as np, pandas as pd, statsmodels.api as sm
from scipy import stats
import sys as _sys; from pathlib import Path; _sys.path.insert(0, str(Path(__file__).resolve().parent))
from regen import *
from statsmodels.stats.diagnostic import breaks_cusumolsresid
which = sys.argv[1]; t0 = time.time()
sd = IDX["NASCDI"].diff().std()

if which == "final":
    t6 = pd.read_csv(TAB / "table6_8_baseline.csv"); rows = []
    for _, r in t6.iterrows():
        spec = {"Symmetric": "level", "SR-symmetric": "sr_sym", "LR-symmetric": "lr_sym", "Unrestricted": "nardl"}[r["final"]]
        o, _ = run(CH[r.chain], IDX, "prod", spec, False, with_sym=False); rows.append(dict(chain=r.chain, final=r["final"], spec=spec, ECT=o["ECT"], lo=o["ECT_lo"], hi=o["ECT_hi"], F=o["F"], F_hi=o["F_hi"], t=o["t"], t_hi=o["t_hi"]))
    pd.DataFrame(rows).to_csv(TAB / "app_final.csv", index=False); print(pd.DataFrame(rows).round(3).to_string())

if which == "a6":
    pct = pd.read_csv(TAB / "tableA8b_percentiles.csv").set_index("percentile")["increase"]; shocks = {"1 SD of weekly change (7.1 pts)": sd, "90th pct weekly increase (11.4 pts)": pct[90], "99th pct weekly increase (27.5 pts)": pct[99]}
    H_ = 26
    def irf(params, names, p, r, s, h):
        g = lambda nm: params[names.index(nm)] if nm in names else 0.0
        lam, gam = g("y_L1"), g("a_L1"); phi = [g(f"dy_L{i}") for i in range(1, p + 1)]; psi = [g(f"da_L{i}") for i in range(r + 1)]
        T = H_ + 1; a = np.zeros(T + 1); a[1:h + 1] = s; y = np.zeros(T + 1); dy = np.zeros(T + 1)
        for t in range(1, T + 1):
            da = lambda k: (a[t - k] - a[t - k - 1]) if t - k >= 1 else 0.0
            e = lam * y[t - 1] + gam * a[t - 1] + sum(phi[i - 1] * (dy[t - i] if t - i >= 1 else 0.0) for i in range(1, p + 1)) + sum(psi[i] * da(i) for i in range(r + 1))
            dy[t] = e; y[t] = y[t - 1] + e
        return 100 * y[1:]
    rows = []
    for ch in CHAINS:
        o, (d, dd, X, m, (p, q, r)) = run(CH[ch], IDX, "prod", "level", False, with_sym=False); names = list(X.columns)
        hac = sm.OLS(dd["dy"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 4, "use_correction": True}); b = hac.params.values; V = hac.cov_params().values
        dr = rng.multivariate_normal(b, V, size=4000); li = names.index("y_L1"); dr = dr[(dr[:, li] < 0) & (dr[:, li] > -1)][:2000]
        for lab0, s in shocks.items():
          for dur in (1, 4):
            lab = f"{lab0}, {dur}-week"
            pt = irf(b, names, p, r, s, dur); D = np.array([irf(x, names, p, r, s, dur) for x in dr]); k = int(np.argmax(np.abs(pt))); lo_, hi_ = np.percentile(D[:, k], [5, 95]); l26, h26 = np.percentile(D[:, -1], [5, 95])
            rows.append(dict(chain=ch, dur=dur, shock=lab, peak=pt[k], peak_wk=k + 1, lo=lo_, hi=hi_, wk26=pt[-1], lo26=l26, hi26=h26))
    pd.DataFrame(rows).to_csv(TAB / "app_a6_peaks.csv", index=False); print(pd.DataFrame(rows).round(2).to_string())

if which == "a7":
    rows = []
    for ch in CHAINS:
        o, (d, dd, X, m, _) = run(CH[ch], IDX, "prod", "nardl", False, with_sym=False); cb = breaks_cusumolsresid(np.asarray(m.resid), ddof=X.shape[1])
        Xv, yv = X.values, dd["dy"].values; n, kk = Xv.shape; wr = []
        for i in range(kk, n):
            bb = np.linalg.lstsq(Xv[:i], yv[:i], rcond=None)[0]; A = np.linalg.pinv(Xv[:i].T @ Xv[:i]); wr.append((yv[i] - Xv[i] @ bb) / np.sqrt(1 + Xv[i] @ A @ Xv[i]))
        wr = np.array(wr); T = len(wr); j = np.arange(1, T + 1); W_ = np.cumsum(wr) / wr.std(ddof=1); bd = 0.948 * (np.sqrt(T) + 2 * j / np.sqrt(T)); breach = bool((np.abs(W_) > bd).any())
        rows.append(dict(chain=ch, n=n, cusum_stat=float(cb[0]), cusum_p=float(cb[1]), bde=("Yes" if breach else "No")))
    pd.DataFrame(rows).to_csv(TAB / "app_a7.csv", index=False); print(pd.DataFrame(rows).round(3).to_string())

if which == "a11":
    rows = []
    for ch in CHAINS:
        out = dict(chain=ch)
        for eq, dep in (("prod", "prod"), ("term", "term")):
            o1, (d, dd, X, m, (p, q, r)) = run(CH[ch], IDX, dep, "nardl", False, with_sym=False); names = list(X.columns); y = dd["dy"].values; Xv = X.values
            sel = [names.index(c) for c in names if c.startswith(("P_L1", "N_L1", "dP_L", "dN_L"))]; Rm = np.zeros((len(sel), len(names)))
            for a, jx in enumerate(sel): Rm[a, jx] = 1
            out[f"{eq}_nardl_p"] = wild_p(y, Xv, Rm)
            o2, (d2, dd2, X2, m2, (p2, q2, r2)) = run(CH[ch], IDX, dep, "level", False, with_sym=False); n2 = list(X2.columns); sel = [n2.index(c) for c in n2 if c.startswith(("a_L1", "da_L"))]; R2 = np.zeros((len(sel), len(n2)))
            for a, jx in enumerate(sel): R2[a, jx] = 1
            out[f"{eq}_level_p"] = wild_p(dd2["dy"].values, X2.values, R2)
            # short-run first-difference regression: dy on da_L0..4, two lags of the other price's change, one own lag
            g = build(CH[ch], IDX, dep, "level"); yv_, xv_ = ("y", "x"); gd = pd.DataFrame(index=g.index); gd["dy"] = g["dy"]; gd["dy1"] = g["dy_L1"]; gd["dx1"] = g["dx_L1"]; gd["dx2"] = g["dx_L2"]; gd["dx0"] = g["dx_L0"]
            for i in range(5): gd[f"da{i}"] = IDX["NASCDI"].diff().shift(i).reindex(g.index)
            gd = gd.dropna(); cols_ = ["dy1", "dx0", "dx1", "dx2"] + [f"da{i}" for i in range(5)]; Xs = sm.add_constant(gd[cols_]).values; ys = gd["dy"].values; nm = ["const"] + cols_
            Rs = np.zeros((5, len(nm)))
            for i in range(5): Rs[i, nm.index(f"da{i}")] = 1
            out[f"{eq}_sr_p"] = wild_p(ys, Xs, Rs); Rsum = np.zeros((1, len(nm)))
            for i in range(5): Rsum[0, nm.index(f"da{i}")] = 1
            b = np.linalg.lstsq(Xs, ys, rcond=None)[0]; est = float((Rsum @ b)[0]); out[f"{eq}_cum_pct"] = 100 * est * sd; out[f"{eq}_cum_p"] = wild_p(ys, Xs, Rsum)
            hac = sm.OLS(ys, Xs).fit(cov_type="HAC", cov_kwds={"maxlags": 4, "use_correction": True}); se = float(np.sqrt((Rsum @ np.asarray(hac.cov_params()) @ Rsum.T).item())); out[f"{eq}_mde"] = 100 * 2.8 * se * sd; out[f"{eq}_n"] = len(gd)
        rows.append(out); print(ch, f"{time.time()-t0:.0f}s", flush=True)
    T = pd.DataFrame(rows); T.to_csv(TAB / "app_a11.csv", index=False); print(T.round(3).to_string())

if which == "a13":
    sc = pd.read_csv(HERE_ := REPO / "data/news_scored/article_scores_v2.csv.gz", parse_dates=["date"]); import sys as _s
    _s.path.insert(0, str(REPO)); from src.nascdi import build_nascdi_final as B
    main_q = (sc["comm"] > 0) & ((sc["d_main"] + sc["m_main"]).clip(-30, 30).abs() >= 1.0); print("main qualifying:", int(main_q.sum()), "of", len(sc))
    s = sc[main_q].copy(); s["clip"] = (s["d_eq"] - s["m_eq"]).clip(-30, 30); s["dscore"] = s["d_eq"]; s["mscore"] = -s["m_eq"]
    daily = s.groupby("date").agg(raw_nascdi=("clip", "mean"), total_score=("clip", "sum"), news_volume=("clip", "count"), disruption_article_count=("clip", lambda x: int((x >= 1).sum())), disruption_score_mean=("dscore", "mean"), mitigation_score_mean=("mscore", "mean")).sort_index(); daily = B.normalize_index(daily, method="z_to_100_10", start_date="2015-02-19", end_date="2025-12-31")
    wk = daily.resample("W").agg(NASCDI=("NASCDI", "mean")).reset_index().rename(columns={"date": "week_end"}).set_index("week_end")
    mainw = pd.read_csv(REPO / "data/nascdi_news_v2/nascdi_news_weekly.csv", parse_dates=["week_end"]).set_index("week_end")["NASCDI"]
    varA = pd.read_csv(REPO / "data/nascdi_news_v2/nascdi_news_weekly_equalweights.csv", parse_dates=["week_end"]).set_index("week_end")["NASCDI"]
    def comp(v):
        j = pd.concat([mainw.rename("m"), v.rename("e")], axis=1).dropna(); dj = j.diff().dropna(); pm, nm_ = load_index(REPO / "data/nascdi_news_v2/nascdi_news_weekly.csv"), None
        pe = np.where(j.e.diff() > 0, j.e.diff(), 0).cumsum(); pmn = np.where(j.m.diff() > 0, j.m.diff(), 0).cumsum(); ne = np.where(j.e.diff() < 0, j.e.diff(), 0).cumsum(); nmn = np.where(j.m.diff() < 0, j.m.diff(), 0).cumsum()
        return dict(levels=j.corr().iloc[0, 1], changes=dj.corr().iloc[0, 1], P=np.corrcoef(pe, pmn)[0, 1], N=np.corrcoef(ne, nmn)[0, 1], floor=(v <= v.min() + 1e-9).mean() * 100)
    A = comp(varA); Bv = comp(wk["NASCDI"]); print("Variant A", A); print("Variant B", Bv); json.dump(dict(A=A, B=Bv, n_main=int(main_q.sum()), n_all=len(sc)), open(TAB / "app_a13.json", "w"), default=float)
print("done", f"{time.time()-t0:.0f}s")
