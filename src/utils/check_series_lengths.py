import glob, os
import pandas as pd

files = sorted(glob.glob("data/model/weekly_model_*_posneg.csv"))
rows = []
for fp in files:
    df = pd.read_csv(fp)
    rows.append({"file": os.path.basename(fp), "n": len(df),
                 "min_week": df["week_end"].min(), "max_week": df["week_end"].max()})
out = pd.DataFrame(rows).sort_values("n")
print(out.to_string(index=False))
print("\nTotal files:", len(out), " | min n:", out["n"].min(), " | max n:", out["n"].max())
