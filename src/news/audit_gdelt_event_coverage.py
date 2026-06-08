import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def load_corpus(news_dir: Path) -> pd.DataFrame:
    files = sorted(news_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {news_dir}")

    frames = []
    for path in files:
        df = pd.read_csv(path, low_memory=False)
        if df.empty or "date" not in df.columns:
            continue
        df["_file"] = path.name
        frames.append(df)
    if not frames:
        raise RuntimeError("No non-empty GDELT CSV files with a date column were found.")

    corpus = pd.concat(frames, ignore_index=True, sort=False)
    corpus["date"] = pd.to_datetime(corpus["date"], errors="coerce").dt.tz_localize(None)
    corpus = corpus.dropna(subset=["date"]).copy()
    for col in ["title", "text", "query_block", "url", "source"]:
        if col not in corpus.columns:
            corpus[col] = ""
        corpus[col] = corpus[col].fillna("").astype(str)
    corpus["full_text"] = (corpus["title"] + " " + corpus["text"]).str.lower()
    corpus["url_norm"] = corpus["url"].str.strip()
    corpus["_fallback_key"] = corpus["date"].dt.strftime("%Y-%m-%d") + "||" + corpus["title"].str.lower().str[:160]
    corpus["_dedupe_key"] = corpus["url_norm"].where(corpus["url_norm"].str.len() > 5, corpus["_fallback_key"])
    corpus = corpus.sort_values("date").drop_duplicates("_dedupe_key", keep="first")
    return corpus


def contains_any(series: pd.Series, values: str) -> pd.Series:
    terms = [term.strip().lower() for term in str(values).split("|") if term.strip()]
    if not terms:
        return pd.Series(True, index=series.index)
    pattern = "|".join(re.escape(term) for term in terms)
    return series.str.contains(pattern, regex=True, na=False)


def audit_windows(corpus: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, event in windows.iterrows():
        start = pd.to_datetime(event["start_date"])
        end = pd.to_datetime(event["end_date"])
        in_window = corpus["date"].between(start, end, inclusive="both")
        block_hit = contains_any(corpus["query_block"].str.lower(), event["required_query_blocks"])
        keyword_hit = contains_any(corpus["full_text"], event["keywords"])
        matches = corpus[in_window & block_hit & keyword_hit].copy()
        minimum = int(event["min_articles"])
        rows.append(
            {
                "event_id": event["event_id"],
                "start_date": start.date().isoformat(),
                "end_date": end.date().isoformat(),
                "description": event.get("description", ""),
                "required_query_blocks": event["required_query_blocks"],
                "keywords": event["keywords"],
                "min_articles": minimum,
                "articles_found": int(len(matches)),
                "unique_sources": int(matches["source"].replace("", pd.NA).nunique(dropna=True)),
                "first_article_date": "" if matches.empty else matches["date"].min().date().isoformat(),
                "last_article_date": "" if matches.empty else matches["date"].max().date().isoformat(),
                "status": "PASS" if len(matches) >= minimum else "FAIL",
            }
        )
    return pd.DataFrame(rows)


def build_block_year_table(corpus: pd.DataFrame) -> pd.DataFrame:
    expanded = corpus.assign(query_block=corpus["query_block"].str.split("|")).explode("query_block")
    expanded["query_block"] = expanded["query_block"].fillna("").str.strip()
    expanded = expanded[expanded["query_block"] != ""].copy()
    expanded["year"] = expanded["date"].dt.year
    return (
        expanded.groupby(["year", "query_block"], as_index=False)
        .agg(articles=("title", "size"), unique_sources=("source", "nunique"))
        .sort_values(["year", "query_block"])
    )


def main():
    parser = argparse.ArgumentParser(
        description="Audit whether the GDELT corpus captures required NASCDI shock windows.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--news_dir", default="data/news/raw_gdelt_news")
    parser.add_argument("--event_windows", default="config/gdelt_event_windows.csv")
    parser.add_argument("--out_dir", default="results_news/coverage_audit")
    parser.add_argument("--fail_on_missing", action="store_true")
    args = parser.parse_args()

    corpus = load_corpus(Path(args.news_dir))
    windows = pd.read_csv(args.event_windows)
    required = {
        "event_id",
        "start_date",
        "end_date",
        "required_query_blocks",
        "keywords",
        "min_articles",
    }
    missing = sorted(required - set(windows.columns))
    if missing:
        raise ValueError(f"Event-window file is missing columns: {missing}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    coverage = audit_windows(corpus, windows)
    block_year = build_block_year_table(corpus)

    coverage_path = out_dir / "major_event_coverage.csv"
    block_year_path = out_dir / "query_block_year_coverage.csv"
    metadata_path = out_dir / "coverage_audit_metadata.json"
    coverage.to_csv(coverage_path, index=False)
    block_year.to_csv(block_year_path, index=False)

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "news_dir": args.news_dir,
        "event_windows": args.event_windows,
        "deduplicated_articles": int(len(corpus)),
        "date_start": corpus["date"].min().date().isoformat(),
        "date_end": corpus["date"].max().date().isoformat(),
        "events_passed": int((coverage["status"] == "PASS").sum()),
        "events_failed": int((coverage["status"] == "FAIL").sum()),
        "outputs": [str(coverage_path), str(block_year_path)],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(coverage[["event_id", "articles_found", "min_articles", "unique_sources", "status"]].to_string(index=False))
    print(f"Saved: {coverage_path}")
    print(f"Saved: {block_year_path}")
    print(f"Saved: {metadata_path}")

    if args.fail_on_missing and (coverage["status"] == "FAIL").any():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
