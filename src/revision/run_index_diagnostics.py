"""Referee 1 #1-#4 and Referee 2 index question: NASCDI coverage, shock distribution, identification, breaks."""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from . import rev_lib as L

ROOT = L.ROOT
DAILY = ROOT / "data" / "nascdi_news" / "nascdi_daily.csv"
END = "2025-03-02"  # end of matched price panel


# ---------------------------------------------------------------- mean-shift DP (Bai-Perron style)
def ssr_matrix(y, h):
    T = len(y)
    cs, cs2 = np.r_[0, np.cumsum(y)], np.r_[0, np.cumsum(y ** 2)]
    i = np.arange(T)[:, None]
    j = np.arange(T)[None, :]
    n = (j - i + 1).astype(float)
    s = cs[j + 1] - cs[i]
    ss = cs2[j + 1] - cs2[i] - s ** 2 / np.where(n > 0, n, 1)
    ss[(j - i + 1) < h] = np.inf
    return ss


def dp_breaks(y, mmax, h):
    T = len(y)
    ss = ssr_matrix(y, h)
    opt = np.full((mmax + 1, T), np.inf)
    arg = np.zeros((mmax + 1, T), int)
    opt[0] = ss[0]
    for m in range(1, mmax + 1):
        for j in range(T):
            cand = opt[m - 1][: j] + ss[1: j + 1, j] if j > 0 else np.array([])
            if len(cand):
                k = int(np.argmin(cand))
                opt[m, j] = cand[k]
                arg[m, j] = k + 1  # start index of last segment
    res = {0: (opt[0][T - 1], [])}
    for m in range(1, mmax + 1):
        brk, j = [], T - 1
        for mm in range(m, 0, -1):
            s = arg[mm, j]
            brk.append(s)
            j = s - 1
        res[m] = (opt[m][T - 1], sorted(brk))
    return res


def supF_next(y, brk, h):
    """sup F(l+1 | l): best additional break inside each of the l+1 segments."""
    T = len(y)
    edges = [0] + list(brk) + [T]
    ss_all = ssr_matrix(y, h)
    ssr_l = sum(ss_all[a, b - 1] for a, b in zip(edges[:-1], edges[1:]))
    best = ssr_l
    for a, b in zip(edges[:-1], edges[1:]):
        if b - a < 2 * h:
            continue
        for s in range(a + h, b - h + 1):
            new = ssr_l - ss_all[a, b - 1] + ss_all[a, s - 1] + ss_all[s, b - 1]
            best = min(best, new)
    l = len(brk)
    return (ssr_l - best) / (best / (T - 2 * (l + 1) - 1)), ssr_l


def seq_tests(y, h, mmax=3, B=299, seed=3):
    rng = np.random.default_rng(seed)
    T = len(y)
    dp = dp_breaks(y, mmax, h)
    out = []
    for l in range(0, mmax):
        brk = dp[l][1]
        stat, _ = supF_next(y, brk, h)
        # null: l breaks + AR(1) errors (residual bootstrap)
        edges = [0] + list(brk) + [T]
        mean = np.concatenate([np.full(b - a, y[a:b].mean()) for a, b in zip(edges[:-1], edges[1:])])
        e = y - mean
        rho = np.corrcoef(e[1:], e[:-1])[0, 1]
        u = e[1:] - rho * e[:-1]
        cnt = 0
        for _ in range(B):
            ub = rng.choice(u, size=T, replace=True)
            eb = np.zeros(T)
            eb[0] = rng.choice(e)
            for t in range(1, T):
                eb[t] = rho * eb[t - 1] + ub[t]
            yb = mean + eb
            dpb = dp_breaks(yb, l, h) if l > 0 else {0: (0, [])}
            sb, _ = supF_next(yb, dpb[l][1], h)
            cnt += sb >= stat
        out.append((l, stat, (cnt + 1) / (B + 1)))
    return dp, out


def info_criteria(y, dp):
    T = len(y)
    rows = []
    for m, (ssr, brk) in dp.items():
        k = 2 * m + 1
        rows.append(dict(breaks=m, SSR=ssr, BIC=T * np.log(ssr / T) + k * np.log(T),
                         LWZ=np.log(ssr / (T - 2 * m - 1)) + 0.299 * (m * 2) * (np.log(T)) ** 2.1 / T,
                         dates=[str(y_idx[b].date()) for b in brk]))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # ---------------- A. coverage by year (daily file)
    d = pd.read_csv(DAILY, parse_dates=["date"])
    d["year"] = d.date.dt.year
    d["no_article"] = d.news_volume == 0
    ydf = d.groupby("year").agg(days=("date", "size"), no_article_days=("no_article", "sum"), articles=("news_volume", "sum"),
                                mean_NASCDI=("NASCDI", "mean"), sd_NASCDI=("NASCDI", "std"))
    ydf["no_article_share"] = ydf.no_article_days / ydf.days
    floor = float(d.loc[d.no_article, "NASCDI"].iloc[0])
    ydf.to_csv(L.OUT_DIR / "index_coverage_by_year.csv")
    print(ydf.round(3).to_string(), "\nfloor value (no-article day):", floor)

    # ---------------- B. weekly series and break analysis
    nas = L.load_nascdi(END)
    z = nas["NASCDI"]
    lv = nas["logvol"]
    y_idx = z.index
    h = int(0.10 * len(z))
    res_tbl = {}
    for nm, ser in (("NASCDI_weekly", z), ("log_news_volume_weekly", lv)):
        y_idx = ser.index
        dp, seq = seq_tests(ser.values.astype(float), h, mmax=3, B=299)
        ic = info_criteria(ser.values.astype(float), dp)
        ic["series"] = nm
        res_tbl[nm] = (ic, seq)
        print(nm)
        print(ic.round(2).to_string())
        print("  sequential supF(l+1|l): ", [(l, round(s, 1), round(p, 3)) for l, s, p in seq])
    pd.concat([v[0] for v in res_tbl.values()]).to_csv(L.OUT_DIR / "break_ic.csv", index=False)
    pd.DataFrame([dict(series=k, l=l, supF=s, p_boot=p) for k, v in res_tbl.items() for l, s, p in v[1]]).to_csv(
        L.OUT_DIR / "break_seq.csv", index=False)

    # ---------------- C. shock distribution
    dz = z.diff().dropna()
    up, dn = dz[dz > 0], dz[dz < 0]
    P, N = L.partial_sums(z)
    pre = dz[dz.index < "2018-03-04"]
    post = dz[dz.index >= "2018-03-04"]
    dist = pd.DataFrame({
        "all": dz.describe(), "pre_2018-03": pre.describe(), "post_2018-03": post.describe()})
    q = pd.DataFrame({"increase_pctile": [50, 75, 90, 95, 99],
                      "dNASCDI_up": [np.percentile(up, p) for p in (50, 75, 90, 95, 99)],
                      "dNASCDI_down": [np.percentile(dn, 100 - p) for p in (50, 75, 90, 95, 99)]})
    dist.to_csv(L.OUT_DIR / "dnascdi_summary.csv")
    q.to_csv(L.OUT_DIR / "dnascdi_percentiles.csv", index=False)
    ps = pd.DataFrame({"mean": [P.mean(), N.mean()], "sd": [P.std(), N.std()], "min": [P.min(), N.min()], "max": [P.max(), N.max()]},
                      index=["NASCDI+", "NASCDI-"])
    ps.to_csv(L.OUT_DIR / "partial_sum_summary.csv")
    print(dist.round(2).to_string(), "\n", q.round(2).to_string(), "\n", ps.round(1).to_string())
    print("weekly dNASCDI: SD", dz.std(), " n up", len(up), " n down", len(dn))

    # ---------------- D. identification diagnostics per chain (on matched estimation weeks)
    rows = []
    for name, path in L.chains():
        f = L.chain_frame(path, nas["NASCDI"]).dropna(subset=["y", "x"])
        t = np.arange(len(f))
        Xv = pd.DataFrame({"P": f.P, "N": f.N, "x": f.x, "yl": f.y})
        vif = {}
        for col in Xv:
            o = [c for c in Xv if c != col]
            vif[col] = 1 / (1 - sm.OLS(Xv[col], sm.add_constant(Xv[o])).fit().rsquared)
        # reparametrised model: Z (level) and P
        Xr = pd.DataFrame({"Z": f.Z, "P": f.P, "x": f.x, "yl": f.y})
        vifr = {}
        for col in Xr:
            o = [c for c in Xr if c != col]
            vifr[col] = 1 / (1 - sm.OLS(Xr[col], sm.add_constant(Xr[o])).fit().rsquared)
        tt = sm.add_constant(np.arange(len(f)))
        rows.append(dict(chain=name, n_weeks=len(f), corr_P_N=np.corrcoef(f.P, f.N)[0, 1],
                         corr_P_time=np.corrcoef(f.P, np.arange(len(f)))[0, 1],
                         R2_P_on_time=sm.OLS(f.P.values, tt).fit().rsquared, R2_N_on_time=sm.OLS(f.N.values, tt).fit().rsquared,
                         VIF_P=vif["P"], VIF_N=vif["N"], VIF_x=vif["x"], VIF_y=vif["yl"],
                         cond_number=np.linalg.cond(sm.add_constant(Xv).values),
                         VIF_P_reparam=vifr["P"], VIF_Z_reparam=vifr["Z"]))
    idt = pd.DataFrame(rows)
    idt.to_csv(L.OUT_DIR / "identification_diagnostics.csv", index=False)
    print(idt.round(3).to_string())
