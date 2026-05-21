import argparse
import hashlib
import json
import platform
import random
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


GDELT_API = "https://api.gdeltproject.org/api/v2/doc/doc"

HEADERS = {
    "User-Agent": "Academic research (NASCDI) - non-commercial; contact: local replication script"
}

# Each block is a complete GDELT query. Keep them broad enough to retrieve
# corridor-disruption news even when articles do not mention apples directly.
QUERY_BLOCKS = {
    "landslide": (
        '(landslide OR landslides OR "shooting stones" OR rockfall OR mudslide) '
        'AND (Kashmir OR Jammu OR Srinagar OR Ramban OR Banihal OR Qazigund OR NH44 OR "NH-44" '
        'OR "Jammu Srinagar National Highway" OR "Jammu-Srinagar National Highway")'
    ),
    "snowfall": (
        '("heavy snowfall" OR snowfall OR snowstorm OR avalanche OR "inclement weather") '
        'AND (Kashmir OR Jammu OR Srinagar OR Ramban OR Banihal OR Qazigund OR NH44 OR "NH-44" '
        'OR "Jammu Srinagar National Highway" OR "Jammu-Srinagar National Highway")'
    ),
    "nh44_updates": (
        '("Jammu Srinagar National Highway" OR "Jammu-Srinagar National Highway" '
        'OR "Srinagar Jammu National Highway" OR "Jammu-Srinagar highway" '
        'OR "Srinagar-Jammu highway" OR "NH 44" OR "NH-44" OR NH44)'
    ),
    "highway_closure": (
        '("highway closed" OR "road closed" OR "traffic suspended" OR "traffic halted" '
        'OR "vehicular movement suspended" OR "tunnel closed" OR "highway reopened" '
        'OR "traffic restored" OR "traffic resumed") '
        'AND (Kashmir OR Jammu OR Srinagar OR Ramban OR Banihal OR Qazigund OR NH44 OR "NH-44" '
        'OR "Jammu Srinagar National Highway" OR "Jammu-Srinagar National Highway")'
    ),
    "market_arrivals": (
        '("low arrivals" OR "arrival shortage" OR "restricted arrivals" OR "delayed consignments" '
        'OR "arrivals improve" OR "market glut" OR "distress sale" OR "distress selling") '
        'AND (apple OR apples OR fruit OR horticulture OR "apple growers" OR "apple crop") '
        'AND (Kashmir OR Srinagar OR Sopore OR Shopian OR Baramulla OR Pulwama OR Azadpur OR Parimpora)'
    ),
    "logistics": (
        '("truck shortage" OR "trucks stranded" OR "stranded trucks" OR "freight disruption" '
        'OR "logistics bottleneck" OR "supply chain disruption" OR "supply restored") '
        'AND (apple OR apples OR fruit OR horticulture OR "apple growers" OR "apple crop" '
        'OR Kashmir OR Srinagar OR Sopore OR Shopian OR Azadpur OR NH44 OR "NH-44")'
    ),
    "political_unrest": (
        '(shutdown OR hartal OR bandh OR curfew OR blockade OR "stone pelting" OR "political unrest" '
        'OR "internet shutdown" OR "communication blackout") '
        'AND (Kashmir OR Jammu OR Srinagar OR Sopore OR Shopian OR NH44 OR "NH-44")'
    ),
    "security": (
        '("terrorist encounter" OR encounter OR "security restrictions" OR "border tension" OR "Indo Pak") '
        'AND (Kashmir OR Jammu OR Srinagar OR Sopore OR Shopian OR NH44 OR "NH-44")'
    ),
    "covid": (
        '(coronavirus OR covid OR lockdown OR "movement restriction") '
        'AND (Kashmir OR Jammu OR Srinagar OR Sopore OR Shopian OR Azadpur OR NH44 OR "NH-44")'
    ),
    "apple_market_general": (
        '(apple OR apples OR "apple growers" OR "apple farmers" OR "apple crop" OR "apple harvest" '
        'OR "fruit growers" OR "fruit mandi") '
        'AND (Kashmir OR Srinagar OR Sopore OR Shopian OR Baramulla OR Pulwama OR Azadpur OR Parimpora)'
    ),
}


def make_session() -> requests.Session:
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        status=3,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    session = requests.Session()
    session.headers.update(HEADERS)
    adapter = HTTPAdapter(max_retries=retry, pool_connections=8, pool_maxsize=8)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def window_ranges(start: datetime, end: datetime, window_days: int) -> Iterable[Tuple[datetime, datetime]]:
    if window_days < 1:
        raise ValueError("--window_days must be at least 1")

    current = start.replace(hour=0, minute=0, second=0, microsecond=0)
    final = end.replace(hour=23, minute=59, second=59, microsecond=0)
    while current <= final:
        window_end = min(current + timedelta(days=window_days) - timedelta(seconds=1), final)
        yield current, window_end
        current = window_end + timedelta(seconds=1)


def add_language_filter(query: str, language: str) -> str:
    if not language:
        return query
    # GDELT allows parentheses around OR groups, but not around a full AND
    # expression. Append query operators without wrapping the whole query.
    return f"{query} sourcelang:{language.lower()}"


def cache_key(query_block: str, query: str, start_dt: datetime, end_dt: datetime, maxrecords: int) -> str:
    raw = f"{query_block}|{query}|{start_dt:%Y-%m-%dT%H:%M:%S}|{end_dt:%Y-%m-%dT%H:%M:%S}|{maxrecords}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def cache_path(
    cache_dir: Path,
    query_block: str,
    query: str,
    start_dt: datetime,
    end_dt: datetime,
    maxrecords: int,
) -> Path:
    return cache_dir / "articles" / f"{cache_key(query_block, query, start_dt, end_dt, maxrecords)}.json"


def gdelt_params(query: str, start_dt: datetime, end_dt: datetime, maxrecords: int, sort: str) -> Dict[str, str]:
    return {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "startdatetime": start_dt.strftime("%Y%m%d%H%M%S"),
        "enddatetime": end_dt.strftime("%Y%m%d%H%M%S"),
        "maxrecords": str(min(maxrecords, 250)),
        "sort": sort,
    }


def fetch(
    session: requests.Session,
    query: str,
    start_dt: datetime,
    end_dt: datetime,
    maxrecords: int,
    timeout: int,
    sort: str,
) -> Dict:
    response = session.get(
        GDELT_API,
        params=gdelt_params(query, start_dt, end_dt, maxrecords, sort),
        timeout=timeout,
    )
    if response.status_code != 200:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text[:300]}")
    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError(f"GDELT did not return JSON: {response.text[:300]}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected GDELT payload type: {type(payload).__name__}")
    return payload


def load_cached_articles(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("articles", [])


def save_cached_payload(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    tmp.replace(path)


def normalize_article(article: Dict, query_block: str, query: str, start_dt: datetime, end_dt: datetime) -> Dict:
    title = article.get("title") or ""
    return {
        "date": article.get("seendate"),
        "title": title,
        "text": article.get("snippet") or "",
        "source": article.get("domain"),
        "url": article.get("url"),
        "query_block": query_block,
        "query": query,
        "window_start": start_dt.strftime("%Y-%m-%d"),
        "window_end": end_dt.strftime("%Y-%m-%d"),
        "source_country": article.get("sourcecountry"),
        "language": article.get("language"),
    }


def write_failure(
    cache_dir: Path,
    query_block: str,
    query: str,
    start_dt: datetime,
    end_dt: datetime,
    maxrecords: int,
    error: Exception,
) -> None:
    fail_path = cache_dir / "failures" / f"{cache_key(query_block, query, start_dt, end_dt, maxrecords)}.txt"
    fail_path.parent.mkdir(parents=True, exist_ok=True)
    fail_path.write_text(
        "\n".join(
            [
                f"query_block={query_block}",
                f"start={start_dt:%Y-%m-%d %H:%M:%S}",
                f"end={end_dt:%Y-%m-%d %H:%M:%S}",
                f"query={query}",
                f"error={error}",
            ]
        ),
        encoding="utf-8",
    )


def fetch_or_cache(
    session: requests.Session,
    query_block: str,
    query: str,
    start_dt: datetime,
    end_dt: datetime,
    cache_dir: Path,
    maxrecords: int,
    retries: int,
    timeout: int,
    sort: str,
) -> List[Dict]:
    path = cache_path(cache_dir, query_block, query, start_dt, end_dt, maxrecords)
    if path.exists():
        return load_cached_articles(path)

    last_error = None
    for attempt in range(1, retries + 1):
        try:
            payload = fetch(session, query, start_dt, end_dt, maxrecords, timeout, sort)
            articles = payload.get("articles", [])
            save_cached_payload(path, {"articles": articles, "query": query})
            return articles
        except Exception as exc:
            last_error = exc
            sleep_for = min(60, (2 ** attempt) + random.random())
            print(f"  retry {attempt}/{retries}: {query_block} {start_dt:%Y-%m-%d} failed: {exc}")
            time.sleep(sleep_for)

    write_failure(cache_dir, query_block, query, start_dt, end_dt, maxrecords, last_error)
    return []


def selected_query_blocks(args: argparse.Namespace) -> List[str]:
    blocks = args.query_block or sorted(QUERY_BLOCKS)
    unknown = sorted(set(blocks) - set(QUERY_BLOCKS))
    if unknown:
        raise ValueError(f"Unknown query blocks: {unknown}")
    return blocks


def dedupe_articles(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.dropna(subset=["date"]).copy()
    df["title"] = df["title"].fillna("").astype(str)
    df = df[df["title"].str.strip() != ""].copy()
    if df.empty:
        return df

    df["url_norm"] = df["url"].fillna("").astype(str).str.strip()
    with_url = df["url_norm"] != ""
    if with_url.any():
        grouped = (
            df[with_url]
            .groupby("url_norm", as_index=False)
            .agg(
                date=("date", "min"),
                title=("title", "first"),
                text=("text", "first"),
                source=("source", "first"),
                url=("url", "first"),
                query_block=("query_block", lambda s: "|".join(sorted(set(s.dropna().astype(str))))),
                query=("query", "first"),
                window_start=("window_start", "min"),
                window_end=("window_end", "max"),
                source_country=("source_country", "first"),
                language=("language", "first"),
            )
        )
        no_url = df[~with_url].drop_duplicates(subset=["date", "title", "query_block"], keep="first")
        df = pd.concat([grouped.drop(columns=["url_norm"], errors="ignore"), no_url], ignore_index=True)
    else:
        df = df.drop_duplicates(subset=["date", "title", "query_block"], keep="first")

    return df.sort_values(["date", "source", "title"], na_position="last")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dataframe_date_range(df: pd.DataFrame, col: str) -> Dict[str, str]:
    if df.empty or col not in df.columns:
        return {"min": "", "max": ""}
    parsed = pd.to_datetime(df[col], errors="coerce")
    return {
        "min": "" if parsed.isna().all() else str(parsed.min()),
        "max": "" if parsed.isna().all() else str(parsed.max()),
    }


def write_run_metadata(out_path: Path, df: pd.DataFrame, args: argparse.Namespace) -> Path:
    meta_path = out_path.with_name(f"{out_path.stem}_metadata.json")
    metadata = {
        "script": "src/news/scrape_gdelt_news.py",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version,
        "platform": platform.platform(),
        "args": vars(args),
        "query_blocks": selected_query_blocks(args),
        "rows_after_dedupe": int(len(df)),
        "date_range": dataframe_date_range(df, "date"),
        "rows_by_query_block": (
            df["query_block"].fillna("").value_counts().sort_index().to_dict()
            if "query_block" in df.columns and not df.empty else {}
        ),
        "rows_by_source_top20": (
            df["source"].fillna("").value_counts().head(20).to_dict()
            if "source" in df.columns and not df.empty else {}
        ),
        "output_csv": str(out_path),
        "output_sha256": file_sha256(out_path) if out_path.exists() else "",
        "note": (
            "GDELT ArtList returns article metadata and titles, not guaranteed full article bodies. "
            "The text column contains snippets only when GDELT provides them."
        ),
    }
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta_path


def build_news_corpus(args: argparse.Namespace) -> pd.DataFrame:
    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d")
    if end < start:
        raise ValueError("--end must be on or after --start")

    if args.maxrecords > 250:
        print("GDELT ArtList caps maxrecords at 250; using 250.")
        args.maxrecords = 250

    blocks = selected_query_blocks(args)
    cache_dir = Path(args.cache_dir)
    session = make_session()
    rows = []

    for query_block in blocks:
        query = add_language_filter(QUERY_BLOCKS[query_block], args.source_lang)
        print(f"\nRunning query block: {query_block}")
        print(f"Query: {query}")

        block_count = 0
        for start_dt, end_dt in window_ranges(start, end, args.window_days):
            articles = fetch_or_cache(
                session=session,
                query_block=query_block,
                query=query,
                start_dt=start_dt,
                end_dt=end_dt,
                cache_dir=cache_dir,
                maxrecords=args.maxrecords,
                retries=args.retries,
                timeout=args.timeout,
                sort=args.sort,
            )
            n = len(articles)
            block_count += n
            if args.verbose or n:
                print(f"  {start_dt:%Y-%m-%d} to {end_dt:%Y-%m-%d}: {n} articles")
            if n >= args.maxrecords:
                print(
                    f"  WARNING: hit maxrecords={args.maxrecords} for {query_block} "
                    f"{start_dt:%Y-%m-%d} to {end_dt:%Y-%m-%d}; reduce --window_days."
                )
            rows.extend(
                normalize_article(article, query_block, query, start_dt, end_dt)
                for article in articles
            )
            time.sleep(random.uniform(args.min_sleep, args.max_sleep))

        print(f"Block total before dedupe: {block_count}")

    df = pd.DataFrame(rows)
    if df.empty:
        if args.allow_empty:
            return pd.DataFrame(columns=["date", "title", "text", "source", "url", "query_block"])
        raise RuntimeError("GDELT returned no articles. Try --window_days 1, broader dates, or fewer filters.")

    df = dedupe_articles(df)
    if df.empty and not args.allow_empty:
        raise RuntimeError("All GDELT rows were empty after cleaning/deduplication.")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a GDELT article-title corpus for news-based NASCDI.")
    parser.add_argument("--start", default="2010-01-01", help="Start date, YYYY-MM-DD.")
    parser.add_argument("--end", default="2025-12-31", help="End date, YYYY-MM-DD.")
    parser.add_argument("--out", default="data/news/raw/news_articles.csv", help="Output CSV path.")
    parser.add_argument("--cache_dir", default="data/news/cache_gdelt", help="Request cache directory.")
    parser.add_argument("--query_block", action="append", choices=sorted(QUERY_BLOCKS), help="Run one query block. Repeat for multiple blocks.")
    parser.add_argument("--maxrecords", type=int, default=250)
    parser.add_argument("--window_days", type=int, default=1, help="Days per GDELT request window. Increase only if no maxrecords warnings appear.")
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--min_sleep", type=float, default=5.5)
    parser.add_argument("--max_sleep", type=float, default=7.5)
    parser.add_argument("--sort", default="DateDesc", choices=["DateDesc", "DateAsc", "HybridRel"], help="GDELT sort order.")
    parser.add_argument("--source_lang", default="english", help="GDELT sourcelang filter. Use empty string to disable.")
    parser.add_argument("--allow_empty", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    df = build_news_corpus(args)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    meta_path = write_run_metadata(out_path, df, args)

    print("\nScraping complete")
    print(f"Articles after dedupe: {len(df)}")
    print(f"Saved to: {out_path}")
    print(f"Metadata: {meta_path}")


if __name__ == "__main__":
    main()
