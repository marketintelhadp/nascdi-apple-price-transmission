import argparse
import hashlib
import json
import os
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


PRICE_PATH = "data/clean/prices_long.csv"
NASCDI_PATH = "data/nascdi/nascdi_event_weekly.csv"
OUT_DIR = "data/model"
MIN_WEEKS_TO_SAVE = 60


def file_sha256(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def csv_row_count(path: Path) -> int:
    if not path.exists() or not path.is_file():
        return 0
    return max(sum(1 for _ in path.open("r", encoding="utf-8", errors="ignore")) - 1, 0)


def safe_filename(value: str) -> str:
    value = str(value).strip().replace(" ", "_")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def to_week_end_sun_midnight(dt: pd.Series) -> pd.Series:
    dt = pd.to_datetime(dt, errors="coerce")
    return dt.dt.to_period("W-SUN").dt.end_time.dt.normalize()


def build_nascdi_posneg(nascdi_weekly: pd.DataFrame) -> pd.DataFrame:
    """
    Construct NASCDI positive and negative partial sums:
      d = change in NASCDI
      NASCDI_pos = cumulative sum of positive changes
      NASCDI_neg = cumulative sum of absolute negative changes
    """
    required = {"week_end", "NASCDI"}
    missing = sorted(required - set(nascdi_weekly.columns))
    if missing:
        raise ValueError(f"NASCDI weekly file is missing columns: {missing}")

    t = nascdi_weekly.copy()
    t = (
        t.sort_values("week_end")
        .drop_duplicates("week_end")
        .reset_index(drop=True)
    )

    t["d_nascdi"] = t["NASCDI"].diff()
    t["NASCDI_pos"] = np.where(t["d_nascdi"] > 0, t["d_nascdi"], 0.0)
    t["NASCDI_neg"] = np.where(t["d_nascdi"] < 0, -t["d_nascdi"], 0.0)
    t["NASCDI_pos"] = t["NASCDI_pos"].cumsum()
    t["NASCDI_neg"] = t["NASCDI_neg"].cumsum()

    return t[["week_end", "NASCDI", "d_nascdi", "NASCDI_pos", "NASCDI_neg"]]


def write_metadata(args, paths, summary_df, merged, skipped_short, out_dir: Path) -> Path:
    output_files = sorted(out_dir.glob("weekly_model_*_posneg.csv"))
    output_files.append(out_dir / "weekly_prices_all_posneg.csv")
    output_files.append(out_dir / "series_summary.csv")

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": "src/data/build_weekly_prices_and_posneg.py",
        "python": sys.version,
        "platform": platform.platform(),
        "arguments": vars(args),
        "inputs": {
            "price_path": {
                "path": str(paths["price_path"]),
                "sha256": file_sha256(paths["price_path"]),
                "rows": csv_row_count(paths["price_path"]),
            },
            "nascdi_path": {
                "path": str(paths["nascdi_path"]),
                "sha256": file_sha256(paths["nascdi_path"]),
                "rows": csv_row_count(paths["nascdi_path"]),
            },
        },
        "outputs": {
            p.name: {
                "path": str(p),
                "sha256": file_sha256(p),
                "rows": csv_row_count(p) if p.suffix.lower() == ".csv" else None,
            }
            for p in output_files
            if p.exists()
        },
        "merge_summary": {
            "master_rows": int(len(merged)),
            "model_series_saved": int(len(summary_df)),
            "series_skipped_short": int(skipped_short),
            "min_weeks_to_save": int(args.min_weeks),
            "week_start": (
                pd.to_datetime(merged["week_end"]).min().date().isoformat()
                if len(merged) else ""
            ),
            "week_end": (
                pd.to_datetime(merged["week_end"]).max().date().isoformat()
                if len(merged) else ""
            ),
        },
    }
    meta_path = out_dir / "model_build_metadata.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    return meta_path


def main():
    parser = argparse.ArgumentParser(
        description="Build weekly price/NASCDI model files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--price_path", default=PRICE_PATH, help="Clean long price CSV.")
    parser.add_argument(
        "--nascdi_path",
        default=NASCDI_PATH,
        help="Weekly NASCDI CSV with date/week_end and NASCDI columns.",
    )
    parser.add_argument("--out_dir", default=OUT_DIR, help="Output directory for model files.")
    parser.add_argument(
        "--min_weeks",
        type=int,
        default=MIN_WEEKS_TO_SAVE,
        help="Minimum observations required to save a per-series model CSV.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    price_path = Path(args.price_path)
    nascdi_path = Path(args.nascdi_path)

    if not price_path.exists():
        raise FileNotFoundError(f"Price file not found: {price_path}")
    if not nascdi_path.exists():
        raise FileNotFoundError(f"NASCDI file not found: {nascdi_path}")

    prices = pd.read_csv(price_path, parse_dates=["date"])
    required_price_cols = {"date", "market_role", "market_name", "variety", "grade", "avg_pk"}
    missing_price_cols = sorted(required_price_cols - set(prices.columns))
    if missing_price_cols:
        raise ValueError(f"Price file is missing columns: {missing_price_cols}")

    if "mask" in prices.columns:
        prices = prices[prices["mask"] == 1].copy()

    prices["week_end"] = to_week_end_sun_midnight(prices["date"])
    prices = prices.dropna(subset=["week_end", "avg_pk"]).copy()

    weekly = (
        prices
        .groupby(["market_role", "market_name", "variety", "grade", "week_end"], as_index=False)
        .agg(avg_price=("avg_pk", "mean"))
    )

    role = weekly["market_role"].astype(str).str.lower()
    prod = weekly[role == "producer"].copy()
    term = weekly[role == "terminal"].copy()
    if prod.empty or term.empty:
        raise RuntimeError("Weekly price data must contain both producer and terminal market_role rows.")

    merged = prod.merge(
        term[["week_end", "variety", "grade", "avg_price", "market_name"]],
        on=["week_end", "variety", "grade"],
        how="inner",
        suffixes=("_prod", "_term"),
    )
    merged = merged.rename(
        columns={
            "market_name_prod": "producer_market",
            "market_name_term": "terminal_market",
            "avg_price_prod": "avg_price_prod",
            "avg_price_term": "avg_price_term",
        }
    )
    merged = merged.dropna(subset=["avg_price_prod", "avg_price_term"]).copy()
    merged = merged[(merged["avg_price_prod"] > 0) & (merged["avg_price_term"] > 0)].copy()
    merged["log_price_prod"] = np.log(merged["avg_price_prod"])
    merged["log_price_term"] = np.log(merged["avg_price_term"])

    nas = pd.read_csv(nascdi_path)
    if "week_end" in nas.columns:
        nas["week_end"] = pd.to_datetime(nas["week_end"], errors="coerce")
    elif "date" in nas.columns:
        nas["week_end"] = pd.to_datetime(nas["date"], errors="coerce")
    else:
        raise ValueError("NASCDI file must contain 'date' or 'week_end'.")
    if "NASCDI" not in nas.columns:
        raise ValueError("NASCDI file must contain a 'NASCDI' column.")

    nas["week_end"] = to_week_end_sun_midnight(nas["week_end"])
    nas = nas[["week_end", "NASCDI"]].dropna().drop_duplicates("week_end").sort_values("week_end")
    nas_posneg = build_nascdi_posneg(nas)

    merged = merged.merge(nas_posneg, on="week_end", how="left")
    matched_rows = int(merged["NASCDI"].notna().sum())
    merged = merged.dropna(subset=["NASCDI", "NASCDI_pos", "NASCDI_neg"]).copy()
    if merged.empty:
        raise RuntimeError("No price rows matched the NASCDI weekly file. Check week_end alignment and dates.")

    master_path = out_dir / "weekly_prices_all_posneg.csv"
    merged.to_csv(master_path, index=False)

    summary_rows = []
    skipped_short = 0
    for (pm, v, g), sub in merged.groupby(["producer_market", "variety", "grade"]):
        sub = sub.sort_values("week_end")
        if len(sub) < args.min_weeks:
            skipped_short += 1
            continue

        fname = f"weekly_model_{safe_filename(pm)}_{safe_filename(v)}_{safe_filename(g)}_posneg.csv"
        out_fp = out_dir / fname
        sub[
            [
                "week_end",
                "avg_price_prod",
                "avg_price_term",
                "log_price_prod",
                "log_price_term",
                "NASCDI",
                "NASCDI_pos",
                "NASCDI_neg",
            ]
        ].to_csv(out_fp, index=False)

        summary_rows.append(
            {
                "producer_market": pm,
                "variety": v,
                "grade": g,
                "rows": int(len(sub)),
                "week_start": pd.to_datetime(sub["week_end"]).min().date().isoformat(),
                "week_end": pd.to_datetime(sub["week_end"]).max().date().isoformat(),
                "file": str(out_fp),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(["producer_market", "variety", "grade"])
    summary_path = out_dir / "series_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    meta_path = write_metadata(
        args=args,
        paths={"price_path": price_path, "nascdi_path": nascdi_path},
        summary_df=summary_df,
        merged=merged,
        skipped_short=skipped_short,
        out_dir=out_dir,
    )

    print(f"Saved master: {master_path} | rows={len(merged)} | matched_rows_before_drop={matched_rows}")
    print(f"Saved per-series model files: {len(summary_df)} | skipped_short={skipped_short}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved metadata: {meta_path}")


if __name__ == "__main__":
    main()
