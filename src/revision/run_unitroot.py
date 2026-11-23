"""Referee 2 #3: unit-root tests in levels AND first differences, with DF-GLS (Elliott-Rothenberg-Stock 1996)."""
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

from . import rev_lib as L

DFGLS_CV = {"1%": -2.58, "5%": -1.95, "10%": -1.62}  # asymptotic, constant only (ERS 1996)


def dfgls(y, maxlag=None):
    y = np.asarray(y, float)
    T = len(y)
    a = 1 - 7.0 / T
    ya = np.r_[y[0], y[1:] - a * y[:-1]]
    za = np.r_[1.0, np.full(T - 1, 1 - a)]
    beta = (za @ ya) / (za @ za)
    yd = y - beta
    if maxlag is None:
        maxlag = int(np.ceil(12 * (T / 100) ** 0.25))
    best = None
    dyd = np.diff(yd)
    for k in range(0, maxlag + 1):
        # common sample for lag choice
        Y = dyd[maxlag:]
        cols = [yd[maxlag:-1]]
        for i in range(1, k + 1):
            cols.append(dyd[maxlag - i:-i if i else None][: len(Y)])
        M = np.column_stack(cols)
        b = np.linalg.lstsq(M, Y, rcond=None)[0]
        e = Y - M @ b
        n = len(Y)
        s2k = e @ e / n
        tau = (b[0] ** 2) * (yd[maxlag:-1] @ yd[maxlag:-1]) / s2k
        maic = np.log(s2k) + 2.0 * (tau + k) / n  # Ng-Perron (2001) modified AIC
        if best is None or maic < best[0]:
            best = (maic, k)
    k = best[1]
    Y = dyd[k:]
    cols = [yd[k:-1]]
    for i in range(1, k + 1):
        cols.append(dyd[k - i:-i][: len(Y)])
    M = np.column_stack(cols)
    b = np.linalg.lstsq(M, Y, rcond=None)[0]
    e = Y - M @ b
    s2 = e @ e / (len(Y) - M.shape[1])
    se = np.sqrt(s2 * np.linalg.inv(M.T @ M)[0, 0])
    return b[0] / se, k


def run_tests(s: pd.Series):
    s = s.dropna()
    adf = adfuller(s, regression="c", autolag="AIC")
    kp = kpss(s, regression="c", nlags="auto")
    g, k = dfgls(s.values)
    return dict(ADF_stat=adf[0], ADF_p=adf[1], KPSS_stat=kp[0], KPSS_p=kp[1], DFGLS_stat=g, DFGLS_lag=k,
                DFGLS_rej5=bool(g < DFGLS_CV["5%"]))


if __name__ == "__main__":
    nas = L.load_nascdi()
    P, N = L.partial_sums(nas["NASCDI"])
    rows = []
    for name, path in L.chains():
        x = pd.read_csv(path, parse_dates=["week_end"]).set_index("week_end").sort_index()
        ser = {"log_price_prod": x["log_price_prod"], "log_price_term": x["log_price_term"],
               "NASCDI": nas["NASCDI"].reindex(x.index), "NASCDI_plus": P.reindex(x.index), "NASCDI_minus": N.reindex(x.index)}
        for v, s in ser.items():
            for order, ss in (("level", s), ("first_diff", s.diff())):
                r = run_tests(ss)
                r.update(chain=name, variable=v, transform=order, n=int(ss.notna().sum()))
                rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(L.OUT_DIR / "unit_root_tests.csv", index=False)
    pd.set_option("display.width", 250)
    summ = out.groupby(["variable", "transform"]).agg(ADF_lo=("ADF_stat", "min"), ADF_hi=("ADF_stat", "max"),
                                                      ADFp_lo=("ADF_p", "min"), ADFp_hi=("ADF_p", "max"),
                                                      KPSS_lo=("KPSS_stat", "min"), KPSS_hi=("KPSS_stat", "max"),
                                                      DFGLS_lo=("DFGLS_stat", "min"), DFGLS_hi=("DFGLS_stat", "max"),
                                                      DFGLS_rej5=("DFGLS_rej5", "sum"))
    summ.to_csv(L.OUT_DIR / "unit_root_summary.csv")
    print(summ.round(3).to_string())
