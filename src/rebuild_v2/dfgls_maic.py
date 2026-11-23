import sys, glob, re
import numpy as np, pandas as pd
from pathlib import Path
import os
REPO = Path(os.environ.get("NASCDI_ROOT", Path(__file__).resolve().parents[2]))

def gls_detrend(y, trend=False):
    T = len(y); cbar = -13.5 if trend else -7.0; a = 1 + cbar / T
    z = np.column_stack([np.ones(T), np.arange(1, T + 1)]) if trend else np.ones((T, 1))
    ys = np.r_[y[0], y[1:] - a * y[:-1]]; zs = np.vstack([z[0], z[1:] - a * z[:-1]])
    b = np.linalg.lstsq(zs, ys, rcond=None)[0]; return y - z @ b

def dfgls_maic(y, trend=False):
    y = np.asarray(y, float); y = y[~np.isnan(y)]; T = len(y); yt = gls_detrend(y, trend)
    kmax = int(12 * (T / 100) ** 0.25); dy = np.diff(yt); ylag = yt[:-1]; s0 = kmax; best = None
    for k in range(kmax + 1):
        X = np.column_stack([ylag[s0:]] + [dy[s0 - j:len(dy) - j] for j in range(1, k + 1)]); Y = dy[s0:]
        b = np.linalg.lstsq(X, Y, rcond=None)[0]; e = Y - X @ b; n = len(Y); s2 = e @ e / n; tau = b[0] ** 2 * (ylag[s0:] @ ylag[s0:]) / s2
        m = np.log(s2) + 2 * (tau + k) / n
        if best is None or m < best[0]: best = (m, k)
    k = best[1]
    X = np.column_stack([ylag[k:]] + [dy[k - j:len(dy) - j] for j in range(1, k + 1)]); Y = dy[k:]
    b = np.linalg.lstsq(X, Y, rcond=None)[0]; e = Y - X @ b; s2 = e @ e / (len(Y) - X.shape[1]); cov = s2 * np.linalg.inv(X.T @ X)
    return b[0] / np.sqrt(cov[0, 0])

def summarize(folder, label):
    rows = []
    for f in sorted(glob.glob(str(REPO / f"data/{folder}/weekly_model_*_posneg.csv"))):
        df = pd.read_csv(f, parse_dates=["week_end"]).sort_values("week_end")
        for var, col in (("NASCDI", "NASCDI"), ("P", "NASCDI_pos"), ("N", "NASCDI_neg"), ("prod", "log_price_prod"), ("term", "log_price_term")):
            s = df[col].values
            rows.append(dict(var=var, form="level", stat=dfgls_maic(s)))
            rows.append(dict(var=var, form="diff", stat=dfgls_maic(np.diff(s))))
    r = pd.DataFrame(rows); out = {}
    for (v, f), g in r.groupby(["var", "form"]): out[(v, f)] = (f"{g.stat.min():.2f} to {g.stat.max():.2f}", f"{int((g.stat < -1.95).sum())} of {len(g)}")
    print(label)
    for k, v in out.items(): print(" ", k, v)
    return out

if __name__ == "__main__":
        summarize("model_news_v2", "NEW")
