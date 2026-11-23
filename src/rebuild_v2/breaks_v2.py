"""Index behaviour on the rebuilt corpus: coverage by year, Bai-Perron-style mean-shift partitions (BIC) and sequential sup-F tests with AR(1) bootstrap,
for the OLD and the NEW weekly NASCDI and for the log weekly record count."""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); warnings.filterwarnings("ignore")
import os
REPO = Path(os.environ.get("NASCDI_ROOT", Path(__file__).resolve().parents[2]))
OUT = REPO / "results_rebuild_v2"; OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(20261120)

old_w = pd.read_csv(REPO / "data/nascdi_news/nascdi_news_weekly.csv", parse_dates=["week_end"]).set_index("week_end")
new_w = pd.read_csv(REPO / "data/nascdi_news_v2/nascdi_news_weekly.csv", parse_dates=["week_end"]).set_index("week_end")
new_d = pd.read_csv(REPO / "data/nascdi_news_v2/nascdi_daily.csv", parse_dates=["date"]).set_index("date")
old_d = pd.read_csv(REPO / "data/nascdi_news/nascdi_daily.csv", parse_dates=["date"]).set_index("date")

print("=" * 100 + "\n1. Coverage by year (daily): days with no qualifying record, records, mean NASCDI, share of weeks at the floor\n" + "=" * 100)
fl_new = new_d.NASCDI.min() + 1e-9; fl_old = old_d.NASCDI.min() + 1e-9
rows = []
for y in range(2015, 2026):
    a = new_d[new_d.index.year == y]; b = old_d[old_d.index.year == y]
    wn = new_w[new_w.index.year == y]; wo = old_w[old_w.index.year == y]
    rows.append(dict(year=y, days=len(a), days_no_record_NEW=int((a.news_volume == 0).sum()), pct_NEW=round((a.news_volume == 0).mean() * 100, 1), records_NEW=int(a.news_volume.sum()),
                     days_no_record_OLD=int((b.news_volume == 0).sum()), records_OLD=int(b.news_volume.sum()), mean_NASCDI_NEW=round(a.NASCDI.mean(), 1),
                     weeks_at_floor_NEW_pct=round((wn.NASCDI <= fl_new).mean() * 100), weeks_at_floor_OLD_pct=round((wo.NASCDI <= fl_old).mean() * 100)))
cov = pd.DataFrame(rows); print(cov.to_string(index=False)); cov.to_csv(OUT / "coverage_by_year.csv", index=False)
print(f"\nNew index: mean {new_w.NASCDI.mean():.2f}, SD {new_w.NASCDI.std():.2f}, weeks at floor {(new_w.NASCDI <= fl_new).mean()*100:.1f}%, range {new_w.NASCDI.min():.2f}-{new_w.NASCDI.max():.2f}")
print(f"Old index: mean {old_w.NASCDI.mean():.2f}, SD {old_w.NASCDI.std():.2f}, weeks at floor {(old_w.NASCDI <= fl_old).mean()*100:.1f}%")
j = new_w.NASCDI.to_frame("new").join(old_w.NASCDI.rename("old"), how="inner")
print(f"Correlation new vs old weekly (all weeks): {j.corr().iloc[0,1]:.3f}; before 2018-03: {j[j.index < '2018-03-01'].corr().iloc[0,1]:.3f}")

# ---------------- mean-shift machinery
def seg_ssr_tables(y):
    n = len(y); c1 = np.r_[0, np.cumsum(y)]; c2 = np.r_[0, np.cumsum(y ** 2)]
    def ssr(i, j):  # segment y[i:j]
        m = j - i; s = c1[j] - c1[i]; return (c2[j] - c2[i]) - s * s / m
    return ssr
def best_partition(y, mmax=3, minlen=None):
    n = len(y); minlen = minlen or max(int(0.10 * n), 5); ssr = seg_ssr_tables(y)
    INF = 1e30
    best = {}   # (k breaks, end index) -> (ssr, breaks)
    # cost[k][j] = min SSR of y[:j] with k breaks
    cost = [np.full(n + 1, INF) for _ in range(mmax + 1)]; arg = [np.zeros(n + 1, int) for _ in range(mmax + 1)]
    for j in range(minlen, n + 1): cost[0][j] = ssr(0, j)
    for k in range(1, mmax + 1):
        for j in range((k + 1) * minlen, n + 1):
            lo, hi = k * minlen, j - minlen
            cands = [(cost[k - 1][i] + ssr(i, j), i) for i in range(lo, hi + 1) if cost[k - 1][i] < INF]
            if cands: v, i = min(cands); cost[k][j] = v; arg[k][j] = i
    out = []
    for k in range(mmax + 1):
        brs = []; j = n; kk = k
        while kk > 0: i = arg[kk][j]; brs.append(i); j = i; kk -= 1
        out.append((cost[k][n], sorted(brs)))
    return out
def sup_f_one(y, trim=0.15):
    n = len(y); ssr = seg_ssr_tables(y); s0 = ssr(0, n); lo, hi = int(trim * n), int((1 - trim) * n); best = (0, None)
    for t in range(lo, hi):
        s1 = ssr(0, t) + ssr(t, n); F = (s0 - s1) / (s1 / (n - 2))
        if F > best[0]: best = (F, t)
    return best
def seq_f(y, brs, minlen):
    """sup-F(l+1|l): best additional break within existing segments."""
    n = len(y); ssr = seg_ssr_tables(y); edges = [0] + list(brs) + [n]; base = sum(ssr(edges[i], edges[i + 1]) for i in range(len(edges) - 1)); best = 0
    for s in range(len(edges) - 1):
        a, b = edges[s], edges[s + 1]
        for t in range(a + minlen, b - minlen + 1):
            new = base - ssr(a, b) + ssr(a, t) + ssr(t, b); F = (base - new) / (new / (n - len(brs) - 2))
            if F > best: best = F
    return best
def ar1_boot(y, fitted, reps=299):
    """AR(1) residual bootstrap around `fitted` (piecewise means)."""
    e = y - fitted; X = np.c_[np.ones(len(e) - 1), e[:-1]]; b = np.linalg.lstsq(X, e[1:], rcond=None)[0]; u = e[1:] - X @ b
    for _ in range(reps):
        us = rng.choice(u, size=len(y), replace=True); es = np.zeros(len(y)); es[0] = e[0]
        for t in range(1, len(y)): es[t] = b[0] + b[1] * es[t - 1] + us[t]
        yield fitted + es

def analyse(name, y, dates, reps=299):
    y = np.asarray(y, float); n = len(y); minlen = max(int(0.10 * n), 5)
    parts = best_partition(y, 3, minlen); res = []
    for k, (s, brs) in enumerate(parts):
        bic = n * np.log(s / n) + (k + 1) * np.log(n) + k * np.log(n)  # mean + k breaks (+ break dates)
        res.append(dict(series=name, breaks=k, SSR=round(s), BIC=round(bic, 1), dates=", ".join(str(dates[i].date()) for i in brs) or "-"))
    print(f"\n{name}"); print(pd.DataFrame(res).to_string(index=False))
    # sequential tests
    out = []
    for k in range(3):
        brs = parts[k][1]; obs = seq_f(y, brs, minlen)
        edges = [0] + list(brs) + [n]; fitted = np.concatenate([np.full(edges[i + 1] - edges[i], y[edges[i]:edges[i + 1]].mean()) for i in range(len(edges) - 1)])
        cnt = 0; B = 0
        for ys in ar1_boot(y, fitted, reps):
            p_ = best_partition(ys, k, minlen)[k][1] if k > 0 else []
            cnt += seq_f(ys, p_, minlen) >= obs; B += 1
        out.append(dict(series=name, test=f"{k} vs {k+1}", supF=round(obs, 1), boot_p=round((cnt + 1) / (B + 1), 3)))
        if (cnt + 1) / (B + 1) > 0.05: break
    print(pd.DataFrame(out).to_string(index=False)); return pd.DataFrame(res), pd.DataFrame(out)

allr = []
for lab, w in (("NEW weekly NASCDI", new_w), ("OLD weekly NASCDI", old_w)):
    r1, r2 = analyse(lab, w.NASCDI.values, w.index); allr += [r1, r2]
for lab, w in (("NEW log weekly record count", new_w), ("OLD log weekly record count", old_w)):
    s = np.log(w.news_volume.clip(lower=1).values); r1, r2 = analyse(lab, s, w.index); allr += [r1, r2]
pd.concat(allr, ignore_index=True).to_csv(OUT / "break_tests.csv", index=False)
