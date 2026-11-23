import sys, glob, re, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss
from arch.unitroot import DFGLS
sys.stdout.reconfigure(encoding="utf-8"); warnings.filterwarnings("ignore")
import os
REPO = Path(os.environ.get("NASCDI_ROOT", Path(__file__).resolve().parents[2]))
OUT = REPO / "results_rebuild_v2" / "unit_roots"; OUT.mkdir(parents=True, exist_ok=True)
FILES = sorted(glob.glob(str(REPO / "data/model_news_v2/weekly_model_*_posneg.csv")))

def gls_detrend(y, trend):
    T = len(y); cbar = -13.5 if trend else -7.0; a = 1 + cbar / T
    z = np.column_stack([np.ones(T), np.arange(1, T + 1)]) if trend else np.ones((T, 1))
    ys = np.r_[y[0], y[1:] - a * y[:-1]]
    zs = np.vstack([z[0], z[1:] - a * z[:-1]])
    b = np.linalg.lstsq(zs, ys, rcond=None)[0]
    return y - z @ b, cbar

def ng_perron(y, trend):
    y = np.asarray(y, float); T = len(y)
    yt, cbar = gls_detrend(y, trend)
    kmax = int(12 * (T / 100) ** 0.25); dy = np.diff(yt); ylag = yt[:-1]
    best = None
    s0 = kmax  # common sample index into dy
    for k in range(kmax + 1):
        X = [ylag[s0:]] + [dy[s0 - j:len(dy) - j] for j in range(1, k + 1)]
        X = np.column_stack(X); Y = dy[s0:]
        b = np.linalg.lstsq(X, Y, rcond=None)[0]; e = Y - X @ b; n = len(Y)
        s2 = e @ e / n; tau = b[0] ** 2 * (ylag[s0:] @ ylag[s0:]) / s2
        maic = np.log(s2) + 2 * (tau + k) / n
        if best is None or maic < best[0]: best = (maic, k)
    k = best[1]
    X = [ylag[k:]] + [dy[k - j:len(dy) - j] for j in range(1, k + 1)]
    X = np.column_stack(X); Y = dy[k:]
    b = np.linalg.lstsq(X, Y, rcond=None)[0]; e = Y - X @ b
    s2e = e @ e / (len(Y) - k - 1); s2 = s2e / (1 - b[1:].sum()) ** 2
    S = (yt[:-1] @ yt[:-1]) / T ** 2
    mza = (yt[-1] ** 2 / T - s2) / (2 * S); msb = np.sqrt(S / s2); mzt = mza * msb
    mpt = (cbar ** 2 * S - cbar * yt[-1] ** 2 / T) / s2 if not trend else (cbar ** 2 * S + (1 - cbar) * yt[-1] ** 2 / T) / s2
    crit = dict(MZa=-17.3, MZt=-2.91, MSB=0.168, MPT=5.48) if trend else dict(MZa=-8.1, MZt=-1.98, MSB=0.233, MPT=3.17)
    rej = (mza < crit["MZa"]) + (mzt < crit["MZt"]) + (msb < crit["MSB"]) + (mpt < crit["MPT"])
    return dict(k=k, MZa=mza, MZt=mzt, MSB=msb, MPT=mpt, NP_rej5=int(rej))

def battery(y, trend):
    y = np.asarray(y, float); y = y[~np.isnan(y)]
    r = dict(n=len(y))
    if np.ptp(y) == 0 or len(y) < 20: return r
    reg = "ct" if trend else "c"
    try:
        a = adfuller(y, regression=reg, autolag="AIC"); r.update(ADF=a[0], ADF_p=a[1], ADF_lag=a[2])
    except Exception: pass
    try:
        kp = kpss(y, regression=reg, nlags="auto"); r.update(KPSS=kp[0], KPSS_p=kp[1])
    except Exception: pass
    try:
        d = DFGLS(y, trend=reg, method="aic"); r.update(DFGLS=d.stat, DFGLS_p=d.pvalue)
    except Exception: pass
    try: r.update(ng_perron(y, trend))
    except Exception as ex: r["NP_err"] = str(ex)[:40]
    return r

def parse(p):
    m = re.match(r"weekly_model_(.+)_(American|Delicious)_(A|B)_posneg", Path(p).stem); return f"{m.group(1)}-{m.group(2)[:3]}-{m.group(3)}"

nas = pd.read_csv(REPO / "data/nascdi_news_v2/nascdi_news_weekly.csv", parse_dates=["week_end"]).sort_values("week_end").drop_duplicates("week_end")
rows = []
def add(chain, var, y, form, trend):
    y = np.asarray(y, float)
    if form == "diff": y = np.diff(y)
    rows.append(dict(chain=chain, var=var, form=form, det="c+t" if trend else "c", **battery(y, trend)))

for fp in FILES:
    ch = parse(fp); df = pd.read_csv(fp, parse_dates=["week_end"]).sort_values("week_end")
    for var, col in (("y_prod", "log_price_prod"), ("x_term", "log_price_term"), ("P", "NASCDI_pos"), ("N", "NASCDI_neg"), ("NASCDI", "NASCDI")):
        for form in ("level", "diff"):
            for trend in ((False, True) if form == "level" else (False,)):
                add(ch, var, df[col].values, form, trend)
for form in ("level", "diff"):
    for trend in ((False, True) if form == "level" else (False,)):
        add("FULL-weekly", "NASCDI", nas["NASCDI"].values, form, trend)
res = pd.DataFrame(rows); res.to_csv(OUT / "unit_root_all.csv", index=False)

# strict within-block first-difference Fisher test
brow = []
for fp in FILES:
    ch = parse(fp); df = pd.read_csv(fp, parse_dates=["week_end"]).sort_values("week_end")
    gap = df["week_end"].diff().dt.days.fillna(7).values > 7
    bid = np.cumsum(gap)
    for var, col in (("y_prod", "log_price_prod"), ("x_term", "log_price_term")):
        ps = []; nb = 0
        for b in np.unique(bid):
            s = df.loc[bid == b, col].values
            d = np.diff(s)
            if len(d) >= 10 and np.ptp(d) > 0:
                try: ps.append(adfuller(d, regression="c", maxlag=0, autolag=None)[1]); nb += 1
                except Exception: pass
        if ps:
            fisher = -2 * np.sum(np.log(np.clip(ps, 1e-300, 1))); p = stats.chi2.sf(fisher, 2 * len(ps))
            brow.append(dict(chain=ch, var=var, blocks_used=nb, fisher_p_diff=p, max_block_p=max(ps)))
        else: brow.append(dict(chain=ch, var=var, blocks_used=0))
pd.DataFrame(brow).to_csv(OUT / "block_diff_fisher.csv", index=False)

pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
f = lambda x: x.round(3)
for form, det in (("diff", "c"), ("level", "c"), ("level", "c+t")):
    sub = res[(res.form == form) & (res.det == det)]
    print(f"\n=== {form.upper()}  det={det} ===")
    print(sub[["chain", "var", "n", "ADF", "ADF_p", "ADF_lag", "KPSS_p", "DFGLS", "DFGLS_p", "k", "MZt", "MPT", "NP_rej5"]].round(3).to_string(index=False))
print("\n=== block-wise first-difference (strict consecutive weeks) ===")
print(pd.DataFrame(brow).round(4).to_string(index=False))
