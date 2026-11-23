"""Chain-level estimation driver used by the revision scripts."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import rev_lib as L


def mark_stale(f: pd.DataFrame, k: int) -> pd.DataFrame:
    """Set producer/terminal prices to NaN inside runs of >= k identical consecutive weekly values
    (keeping the first week of each run), i.e. treat carried-forward Agmarknet reports as missing."""
    if not k:
        return f
    f = f.copy()
    for col, logcol in (("pprod", "y"), ("pterm", "x")):
        if col not in f:
            continue
        v = f[col]
        same = (v == v.shift(1)) & v.notna() & v.shift(1).notna()
        run = same.groupby((~same).cumsum()).cumsum()  # position within run of repeats
        # a run of repeats of length m corresponds to m+1 identical values
        run_len = same.groupby((~same).cumsum()).transform("sum") + 1
        stale = same & (run_len >= k)
        f.loc[stale, logcol] = np.nan
    return f


def estimate_chain(name, path, index, label="base", start=None, stale=0, dep="prod", controls=None,
                   cv_reps=3000, boot=1999, extra=None, lags=None, select_rule="aic", verbose=False, end=None):
    ex = dict(extra or {})
    f = L.chain_frame(path, index, ex)
    raw = pd.read_csv(path, parse_dates=["week_end"]).set_index("week_end")
    f["pterm"] = raw["avg_price_term"].reindex(f.index)
    f = mark_stale(f, stale)
    if start is not None:
        f = f.loc[pd.Timestamp(start):]
    if end is not None:
        f = f.loc[:pd.Timestamp(end)]
    out = {"chain": name, "variant": label}
    if lags is None:
        p, q, r, grid, flagged = L.select_lags(f, dep=dep, ic=select_rule, controls=controls)
    else:
        (p, q, r), flagged, grid = lags, False, None
    out.update(p=p, q=q, r=r, bg_flag=flagged)

    # --- unrestricted NARDL (asymmetric)
    d, y, X, plain, hac = L.fit_chain(f, p, q, r, "asym", dep, controls)
    names = X.columns.tolist()
    n = len(y)
    out.update(n=n, k_params=X.shape[1], n_grid_weeks=int(f["y"].notna().sum()))
    a = plain.params["y_L1"]
    out.update(ECT=a, ECT_se_plain=plain.bse["y_L1"], ECT_p_plain=plain.pvalues["y_L1"],
               ECT_se_hac=hac.bse["y_L1"], ECT_p_hac=hac.pvalues["y_L1"],
               half_life=L.half_life(a), R2=plain.rsquared, AIC=plain.aic, BIC=plain.bic)
    ci = hac.conf_int(alpha=0.05).loc["y_L1"].values
    out.update(ECT_lo=ci[0], ECT_hi=ci[1])
    b = plain.params
    out.update(LR_x=-b["x_L1"] / a, LR_P=-b["P_L1"] / a, LR_N=-b["N_L1"] / a)
    F, t, kk = L.bounds_stats(plain, names, "asym")
    cv = L.sim_bounds_cv(n, p, q, r, kk, reps=cv_reps, seed=11)
    out.update(F=F, t=t, F_L5=cv["L"]["F05"], F_U5=cv["U"]["F05"], F_U10=cv["U"]["F10"], F_U1=cv["U"]["F01"],
               t_L5=cv["L"]["t05"], t_U5=cv["U"]["t05"], t_U10=cv["U"]["t10"])
    out["bounds_F_decision5"] = "coint" if F > cv["U"]["F05"] else ("none" if F < cv["L"]["F05"] else "inconclusive")
    out["bounds_t_decision5"] = "coint" if t < cv["U"]["t05"] else ("none" if t > cv["L"]["t05"] else "inconclusive")
    out["bounds_F_decision10"] = "coint" if F > cv["U"]["F10"] else ("none" if F < cv["L"]["F10"] else "inconclusive")

    # --- symmetry tests
    lr, sr, jt = L.restr_matrices(names)
    for tag, R in (("LR", lr), ("SR", sr), ("JT", jt)):
        W, pv = L.wald(hac, R)
        out[f"{tag}_wald_hac"], out[f"{tag}_p_hac"] = W, pv
        if boot:
            W3, pb = L.wild_boot_wald(y, X, R, B=boot, seed=5)
            out[f"{tag}_wald_hc3"], out[f"{tag}_p_boot"] = W3, pb
    # --- diagnostics (unrestricted model, plain residuals)
    out.update({f"diag_{k}": v for k, v in L.diagnostics(plain, d, X).items()})

    # --- final restricted model: impose the restrictions not rejected at 5% (wild bootstrap p)
    pLR = out.get("LR_p_boot", out["LR_p_hac"])
    pSR = out.get("SR_p_boot", out["SR_p_hac"])
    lr_ok, sr_ok = pLR > 0.05, pSR > 0.05
    spec = {(True, True): "sym", (True, False): "lr_sym", (False, True): "sr_sym", (False, False): "asym"}[(lr_ok, sr_ok)]
    out["final_spec"] = spec
    d2, y2, X2, plain2, hac2 = L.fit_chain(f, p, q, r, spec, dep, controls)
    n2 = len(y2)
    a2 = plain2.params["y_L1"]
    out.update(n_final=n2, ECT_final=a2, ECT_final_se_hac=hac2.bse["y_L1"], ECT_final_p_hac=hac2.pvalues["y_L1"],
               half_life_final=L.half_life(a2), R2_final=plain2.rsquared, AIC_final=plain2.aic)
    ci2 = hac2.conf_int(alpha=0.05).loc["y_L1"].values
    out.update(ECT_final_lo=ci2[0], ECT_final_hi=ci2[1])
    b2 = plain2.params
    out["LR_x_final"] = -b2["x_L1"] / a2
    if "Z_L1" in b2:
        out["LR_Z_final"] = -b2["Z_L1"] / a2
    F2, t2, k2 = L.bounds_stats(plain2, X2.columns.tolist(), spec)
    cv2 = L.sim_bounds_cv(n2, p, q, r, k2, reps=cv_reps, seed=13)
    out.update(F_final=F2, t_final=t2, F_final_U5=cv2["U"]["F05"], F_final_L5=cv2["L"]["F05"],
               t_final_U5=cv2["U"]["t05"], t_final_L5=cv2["L"]["t05"])
    out["final_F_decision5"] = "coint" if F2 > cv2["U"]["F05"] else ("none" if F2 < cv2["L"]["F05"] else "inconclusive")
    out["final_t_decision5"] = "coint" if t2 < cv2["U"]["t05"] else ("none" if t2 > cv2["L"]["t05"] else "inconclusive")
    out.update({f"final_diag_{k}": v for k, v in L.diagnostics(plain2, d2, X2).items()})
    return out, (f, p, q, r)
