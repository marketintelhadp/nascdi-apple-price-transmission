import os
import pandas as pd

DATA = "data/model/weekly_prices_all_posneg.csv"
OUT = "results/nardl"
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(DATA)

rows = []

for (m, v, g), sub in df.groupby(
    ["producer_market", "variety", "grade"]
):
    rows.append({
        "Market": m,
        "Variety": v,
        "Grade": g,
        "Producer_mean": sub["avg_price_prod"].mean(),
        "Producer_sd": sub["avg_price_prod"].std(),
        "Terminal_mean": sub["avg_price_term"].mean(),
        "Terminal_sd": sub["avg_price_term"].std(),
        "Observations": len(sub)
    })

table1 = pd.DataFrame(rows)

table1 = table1.sort_values(
    ["Market", "Variety", "Grade"]
)

out_fp = os.path.join(OUT, "table1_descriptive_stats.csv")
table1.to_csv(out_fp, index=False)

print("✅ Table 1 saved:", out_fp)
