import json, numpy as np, pandas as pd
from regen import *
from scipy import stats
out = {}
t6 = pd.read_csv(TAB / "table6_8_baseline.csv"); A9 = pd.read_csv(TAB / "tableA9_bounds_both_directions.csv")
# 1 final specifications
fin = {}
for _, r in t6.iterrows():
    spec = {"Symmetric": "level", "SR-symmetric": "sr_sym", "LR-symmetric": "lr_sym", "Unrestricted": "nardl"}[r["final"]]
    o, _ = run(CH[r.chain], IDX, "prod", spec, False, with_sym=False); fin[r.chain] = dict(spec=r["final"], ECT=o["ECT"], half=o["half_life"], n=o["n"], F=o["F"], t=o["t"], cat=o["cat"])
out["final"] = fin
sh = [c for c in CHAINS if c.startswith("Shopian")]; so = [c for c in CHAINS if c.startswith("Sopore")]
ect = dict(zip(t6.chain, t6.ECT)); out["meanECT_sh"] = np.mean([ect[c] for c in sh]); out["meanECT_so"] = np.mean([ect[c] for c in so])
fe = {c: fin[c]["ECT"] for c in CHAINS}; out["meanFinal_sh"] = np.mean([fe[c] for c in sh]); out["meanFinal_so"] = np.mean([fe[c] for c in so])
vals = np.array([ect[c] for c in CHAINS]); obs = abs(vals[:4].mean() - vals[4:].mean()); import itertools
perm = [abs(vals[list(i)].mean() - np.delete(vals, list(i)).mean()) for i in itertools.combinations(range(8), 4)]; out["perm_p"] = float(np.mean(np.array(perm) >= obs - 1e-12))
# 2 A9 bootstrap p for baseline equations and level/prices entries
out["A9_prod_nardl"] = A9[(A9.dep == "prod") & (A9.spec == "nardl")][["chain", "t_pboot"]].round(3).to_dict("records")
out["A9_ShDelA"] = A9[(A9.chain == "Shopian-Delicious-A") & (A9.dep == "prod")][["spec", "F", "F_hi", "t", "t_hi", "cat"]].round(2).to_dict("records")
out["A9_SopAmB"] = A9[(A9.chain == "Sopore-American-B") & (A9.dep == "prod")][["spec", "F", "F_hi", "t", "t_hi", "cat"]].round(2).to_dict("records")
# 3 matched-week stats for the index
ws = sorted(set().union(*[set(CH[c].week_end) for c in CHAINS])); m = IDX.reindex(ws)
out["matched264"] = dict(n=len(m), mean=m.NASCDI.mean(), sd=m.NASCDI.std(), mn=m.NASCDI.min(), mx=m.NASCDI.max(), Pm=m.P.mean(), Ps=m.P.std(), Pmax=m.P.max(), Nm=m.N.mean(), Ns=m.N.std(), Nmin=m.N.min())
allw = pd.concat([CH[c][["week_end"]] for c in CHAINS]); m2 = IDX.reindex(allw.week_end); out["pooled1276"] = dict(mean=m2.NASCDI.mean(), sd=m2.NASCDI.std())
# 4 ECT SE ratios
a4 = pd.read_csv(TAB / "tableA4_stability_manuscript_design.csv"); out["se_ratio_max_pct"] = float(((a4.se_HAC / a4.se_plain - 1).abs() * 100).max()); out["se_hc3_ratio_max_pct"] = float(((a4.se_HC3 / a4.se_plain - 1).abs() * 100).max())
# 5 A6 summaries
a6 = pd.read_csv(TAB / "tableA6_dynamic_responses.csv")
for dur in (1, 4):
    s = a6[(a6.shock == "90th pct") & (a6.duration_wks == dur)]; out[f"A6_90_dur{dur}"] = dict(min=s.response_pct.min(), max=s.response_pct.max(), wk4=[s[s.week == 4].response_pct.min(), s[s.week == 4].response_pct.max()], zero_share=s.band_incl_zero.mean(),
        excl=s[~s.band_incl_zero][["chain", "week", "response_pct", "lo90", "hi90"]].round(2).to_dict("records"))
# 6 recursive CUSUM crossing (Brown-Durbin-Evans bounds, a=0.948)
cus = {}
for ch in CHAINS:
    o, (d, dd, X, mm, _) = run(CH[ch], IDX, "prod", "nardl", False, with_sym=False); Xv, yv = X.values, dd["dy"].values; n, k = Xv.shape; w = []
    for i in range(k, n):
        b = np.linalg.lstsq(Xv[:i], yv[:i], rcond=None)[0]; A = np.linalg.pinv(Xv[:i].T @ Xv[:i]); w.append((yv[i] - Xv[i] @ b) / np.sqrt(1 + Xv[i] @ A @ Xv[i]))
    w = np.array(w); s = w.std(ddof=1); W = np.cumsum(w) / s; T = len(w); j = np.arange(1, T + 1); bound = 0.948 * (np.sqrt(T) + 2 * j / np.sqrt(T))
    cus[ch] = dict(breach=bool((np.abs(W) > bound).any()), maxratio=float((np.abs(W) / bound).max()))
out["cusum"] = cus
# 7 A13
out["A13"] = pd.read_csv(TAB / "tableA13_equal_weights.csv").iloc[0].to_dict()
# 8 repeated runs (A10) and A11 rows
out["A10"] = pd.read_csv(TAB / "tableA10_repeated_run_filter.csv").round(3).to_dict("records")
json.dump(out, open("v5_numbers.json", "w"), indent=1, default=lambda o: float(o) if hasattr(o, "item") else str(o)); print("saved")
for k in ("final", "meanECT_sh", "meanECT_so", "meanFinal_sh", "meanFinal_so", "perm_p", "A9_prod_nardl", "A9_ShDelA", "A9_SopAmB", "matched264", "pooled1276", "se_ratio_max_pct", "se_hc3_ratio_max_pct", "A6_90_dur1", "A6_90_dur4", "cusum", "A13"): print(k, out[k])
