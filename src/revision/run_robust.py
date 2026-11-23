"""Robustness variants motivated by Referee 1 #1/#4 (coverage break) and #3 / Table A7 (dependent-variable reversal)."""
import sys
import numpy as np
import pandas as pd

from . import rev_lib as L
from .rev_run import estimate_chain

BREAK = pd.Timestamp("2018-03-04")  # weekly NASCDI mean shift (and weekly record-count shift: 2018-03-11)


def regime_standardised(z: pd.Series) -> pd.Series:
    out = z.copy()
    for m in (z.index < BREAK, z.index >= BREAK):
        s = z[m]
        out[m] = 100 + 10 * (s - s.mean()) / s.std(ddof=0)
    return out


if __name__ == "__main__":
    variant = sys.argv[1]
    nas = L.load_nascdi()
    z = nas["NASCDI"]
    kw = dict(cv_reps=2000, boot=999)
    rows = []
    for name, path in L.chains():
        try:
            if variant == "regime_z":
                res, _ = estimate_chain(name, path, regime_standardised(z), label=variant, **kw)
            elif variant == "post2018":
                res, _ = estimate_chain(name, path, z, label=variant, start=BREAK, **kw)
            elif variant == "pre2018":
                if name.startswith("Shopian"):
                    continue
                res, _ = estimate_chain(name, path, z, label=variant, end=BREAK - pd.Timedelta(days=1), **kw)
            elif variant == "covid":
                covid = pd.Series(((z.index >= "2020-03-01") & (z.index <= "2021-12-31")).astype(float), index=z.index)
                res, _ = estimate_chain(name, path, z, label=variant, controls=["covid"], extra={"covid": covid}, **kw)
            elif variant == "reverse":
                res, _ = estimate_chain(name, path, z, label=variant, dep="term", **kw)
            else:
                raise SystemExit("unknown variant")
        except Exception as e:  # sample too short
            print(name, variant, "FAILED:", e, flush=True)
            continue
        rows.append(res)
        print(name, variant, "n", res["n"], "lags", (res["p"], res["q"], res["r"]), "ECT", round(res["ECT"], 3),
              "F", round(res["F"], 2), res["bounds_F_decision5"], res["bounds_t_decision5"],
              "LRp", round(res["LR_p_boot"], 3), "SRp", round(res["SR_p_boot"], 3), flush=True)
    pd.DataFrame(rows).to_csv(L.OUT_DIR / f"main_{variant}.csv", index=False)
