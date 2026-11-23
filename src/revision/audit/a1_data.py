import sys, importlib.util, numpy as np, pandas as pd
spec = importlib.util.spec_from_file_location("rev_lib", __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "rev_lib.py"))
L = importlib.util.module_from_spec(spec); spec.loader.exec_module(L)
nas = L.load_nascdi()
print("NASCDI weekday set:", nas.index.dayofweek.value_counts().to_dict(), len(nas), nas.index.min(), nas.index.max())
for name, path in L.chains():
    raw = pd.read_csv(path, parse_dates=["week_end"]).sort_values("week_end")
    dow = raw.week_end.dt.dayofweek.value_counts().to_dict()
    f = L.chain_frame(path, nas["NASCDI"])
    gaps = raw.week_end.diff().dt.days.div(7).dropna()
    print(name, "rows", len(raw), "dow", dow, "grid notna y", f.y.notna().sum(), "x", f.x.notna().sum(), "Z", f.Z.notna().sum(),
          "gaps>1wk:", int((gaps>1).sum()), "max gap", int(gaps.max()), "dup", raw.week_end.duplicated().sum())
print(nas.columns.tolist())
