import pandas as pd, glob
files = glob.glob("data/model/weekly_model_*.csv")
for f in files:
    df = pd.read_csv(f)
    if "NASCDI" in df.columns:
        print(f, df["NASCDI"].isna().mean().round(3), df["NASCDI"].nunique())