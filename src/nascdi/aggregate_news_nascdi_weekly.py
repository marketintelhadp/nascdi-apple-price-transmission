import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate news-based daily NASCDI to weekly frequency.")
    parser.add_argument("--daily", default="data/nascdi/nascdi_daily.csv", help="Daily news NASCDI CSV.")
    parser.add_argument("--out", default="data/nascdi/nascdi_news_weekly.csv", help="Weekly output CSV.")
    parser.add_argument("--week_freq", default="W-SUN", help="Pandas weekly frequency. Default is W-SUN.")
    args = parser.parse_args()

    daily_path = Path(args.daily)
    if not daily_path.exists():
        raise FileNotFoundError(f"Daily NASCDI file not found: {daily_path}")

    df = pd.read_csv(daily_path, parse_dates=["date"])
    if df.empty:
        raise ValueError(f"Daily NASCDI is empty: {daily_path}")
    if "NASCDI" not in df.columns:
        raise ValueError("Daily NASCDI file must contain a NASCDI column.")

    df = df.dropna(subset=["date", "NASCDI"]).set_index("date").sort_index()
    if df.empty:
        raise ValueError("No valid dated NASCDI observations remain after cleaning.")

    agg_cols = {
        "raw_nascdi": "mean",
        "total_score": "sum",
        "news_volume": "sum",
        "disruption_article_count": "sum",
        "NASCDI": "mean",
    }
    agg_cols = {col: agg for col, agg in agg_cols.items() if col in df.columns}

    weekly = df.resample(args.week_freq).agg(agg_cols).reset_index()
    weekly = weekly.rename(columns={"date": "week_end"})

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    weekly.to_csv(out_path, index=False)

    print(f"Weekly news NASCDI saved: {out_path}")
    print(f"Rows: {len(weekly)}")


if __name__ == "__main__":
    main()
