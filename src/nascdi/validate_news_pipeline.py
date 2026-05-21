import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def csv_rows(path: Path) -> int:
    if not path.exists() or not path.is_file():
        return 0
    return max(sum(1 for _ in path.open("r", encoding="utf-8", errors="ignore")) - 1, 0)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def add_check(checks, name, status, detail):
    checks.append({"name": name, "status": status, "detail": detail})


def parse_date_column(df: pd.DataFrame, candidates):
    for col in candidates:
        if col in df.columns:
            return col, pd.to_datetime(df[col], errors="coerce")
    return "", pd.Series(dtype="datetime64[ns]")


def check_continuous_daily(dates: pd.Series):
    dates = pd.to_datetime(dates, errors="coerce").dropna().dt.normalize().drop_duplicates().sort_values()
    if dates.empty:
        return False, "no valid dates"
    expected = pd.date_range(dates.min(), dates.max(), freq="D")
    missing = expected.difference(dates)
    if len(missing):
        sample = ", ".join(d.strftime("%Y-%m-%d") for d in missing[:5])
        return False, f"{len(missing)} missing daily dates; first missing: {sample}"
    return True, f"{dates.min().date()} to {dates.max().date()} ({len(dates)} days)"


def check_continuous_weekly(dates: pd.Series):
    dates = pd.to_datetime(dates, errors="coerce").dropna().dt.normalize().drop_duplicates().sort_values()
    if dates.empty:
        return False, "no valid week dates"
    diffs = dates.diff().dropna().dt.days
    bad = diffs[diffs != 7]
    if len(bad):
        return False, f"{len(bad)} non-7-day gaps in weekly index"
    return True, f"{dates.min().date()} to {dates.max().date()} ({len(dates)} weeks)"


def validate(args):
    checks = []
    news_dir = Path(args.news_dir)
    nascdi_dir = Path(args.nascdi_dir)
    model_dir = Path(args.model_dir)
    results_dir = Path(args.results_dir) if args.results_dir else None

    raw_files = sorted(news_dir.glob("*.csv")) if news_dir.exists() else []
    raw_rows = sum(csv_rows(p) for p in raw_files)
    if raw_rows >= args.min_raw_articles:
        add_check(checks, "raw_news_corpus", "PASS", f"{len(raw_files)} CSV files, {raw_rows} rows")
    else:
        add_check(checks, "raw_news_corpus", "FAIL", f"{len(raw_files)} CSV files, {raw_rows} rows")

    scored_path = nascdi_dir / "news_scored.csv"
    scored = read_csv(scored_path)
    if len(scored) > 0:
        add_check(checks, "news_scored", "PASS", f"{len(scored)} scored rows")
    else:
        add_check(checks, "news_scored", "FAIL", f"missing or empty: {scored_path}")

    daily_path = nascdi_dir / "nascdi_daily.csv"
    daily = read_csv(daily_path)
    if len(daily) > 0 and "NASCDI" in daily.columns:
        date_col, dates = parse_date_column(daily, ["date", "week_end"])
        ok, detail = check_continuous_daily(dates)
        add_check(checks, "daily_nascdi_continuity", "PASS" if ok else "FAIL", detail)
        if daily["NASCDI"].isna().any():
            add_check(checks, "daily_nascdi_values", "FAIL", "NASCDI contains missing values")
        elif daily["NASCDI"].nunique(dropna=True) < args.min_unique_nascdi:
            add_check(
                checks,
                "daily_nascdi_values",
                "WARN",
                f"only {daily['NASCDI'].nunique(dropna=True)} unique NASCDI values",
            )
        else:
            add_check(
                checks,
                "daily_nascdi_values",
                "PASS",
                f"mean={daily['NASCDI'].mean():.3f}, std={daily['NASCDI'].std(ddof=1):.3f}, unique={daily['NASCDI'].nunique(dropna=True)}",
            )
        if args.start and args.end and date_col:
            expected = len(pd.date_range(args.start, args.end, freq="D"))
            status = "PASS" if len(daily) == expected else "WARN"
            add_check(checks, "daily_expected_rows", status, f"observed={len(daily)}, expected={expected}")
    else:
        add_check(checks, "daily_nascdi", "FAIL", f"missing, empty, or no NASCDI column: {daily_path}")

    weekly_path = nascdi_dir / "nascdi_news_weekly.csv"
    weekly = read_csv(weekly_path)
    if len(weekly) >= args.min_weekly_rows and "NASCDI" in weekly.columns:
        _, week_dates = parse_date_column(weekly, ["week_end", "date"])
        ok, detail = check_continuous_weekly(week_dates)
        add_check(checks, "weekly_nascdi", "PASS" if ok else "WARN", detail)
    else:
        add_check(checks, "weekly_nascdi", "FAIL", f"rows={len(weekly)}, required={args.min_weekly_rows}")

    for fname in ["nascdi_build_metadata.json", "table1_nascdi_descriptive.csv"]:
        path = nascdi_dir / fname
        add_check(checks, fname, "PASS" if path.exists() else "WARN", str(path))

    master_path = model_dir / "weekly_prices_all_posneg.csv"
    master_rows = csv_rows(master_path)
    if master_rows > 0:
        add_check(checks, "model_master", "PASS", f"{master_rows} rows")
    else:
        add_check(checks, "model_master", "FAIL", f"missing or empty: {master_path}")

    model_files = sorted(model_dir.glob("weekly_model_*_posneg.csv"))
    if len(model_files) >= args.min_model_series:
        add_check(checks, "model_series_files", "PASS", f"{len(model_files)} series files")
    else:
        add_check(checks, "model_series_files", "FAIL", f"{len(model_files)} series files, required={args.min_model_series}")

    for fname in ["series_summary.csv", "model_build_metadata.json"]:
        path = model_dir / fname
        add_check(checks, fname, "PASS" if path.exists() else "WARN", str(path))

    if results_dir:
        result_path = results_dir / "nardl_results_log.csv"
        diag_path = results_dir / "nardl_diagnostics_log.csv"
        meta_path = results_dir / "nardl_run_metadata.json"
        add_check(checks, "nardl_results", "PASS" if csv_rows(result_path) > 0 else "WARN", str(result_path))
        add_check(checks, "nardl_diagnostics", "PASS" if csv_rows(diag_path) > 0 else "WARN", str(diag_path))
        add_check(checks, "nardl_metadata", "PASS" if meta_path.exists() else "WARN", str(meta_path))

    return checks


def write_reports(checks, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "summary": {
            "pass": sum(1 for c in checks if c["status"] == "PASS"),
            "warn": sum(1 for c in checks if c["status"] == "WARN"),
            "fail": sum(1 for c in checks if c["status"] == "FAIL"),
        },
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    md_path = out_path.with_suffix(".md")
    lines = [
        "# News NASCDI Pipeline Validation",
        "",
        f"Created UTC: {payload['created_utc']}",
        "",
        f"PASS: {payload['summary']['pass']} | WARN: {payload['summary']['warn']} | FAIL: {payload['summary']['fail']}",
        "",
        "| Check | Status | Detail |",
        "|---|---:|---|",
    ]
    for check in checks:
        detail = str(check["detail"]).replace("|", "\\|")
        lines.append(f"| {check['name']} | {check['status']} | {detail} |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path


def main():
    parser = argparse.ArgumentParser(
        description="Validate the news/lexicon NASCDI pipeline outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--news_dir", default="data/news/raw_gdelt_news")
    parser.add_argument("--nascdi_dir", default="data/nascdi_news")
    parser.add_argument("--model_dir", default="data/model_news")
    parser.add_argument("--results_dir", default="results_news/nardl")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--min_raw_articles", type=int, default=1)
    parser.add_argument("--min_weekly_rows", type=int, default=10)
    parser.add_argument("--min_model_series", type=int, default=8)
    parser.add_argument("--min_unique_nascdi", type=int, default=10)
    parser.add_argument("--out", default="results_news/news_pipeline_validation.json")
    args = parser.parse_args()

    checks = validate(args)
    report_md = write_reports(checks, Path(args.out))
    for check in checks:
        print(f"{check['status']}: {check['name']} - {check['detail']}")
    print(f"Saved validation report: {args.out}")
    print(f"Saved markdown report: {report_md}")

    if any(check["status"] == "FAIL" for check in checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
