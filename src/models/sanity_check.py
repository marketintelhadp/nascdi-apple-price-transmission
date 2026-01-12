import pandas as pd

df = pd.read_csv(
    "data/model/weekly_prices_all_posneg.csv",
    parse_dates=["week_end"]
)

sub = df[
    (df["producer_market"] == "Sopore") &
    (df["variety"] == "American") &
    (df["grade"] == "A")
]

print(sub[["avg_price_prod", "avg_price_term"]].corr())
print(sub[["avg_price_prod", "avg_price_term"]].describe())

key = ["week_end", "variety", "grade", "producer_market"]
print("Duplicates:", sub.duplicated(key).sum())
