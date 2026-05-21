import argparse
import hashlib
import json
import os
import random
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd
import requests


GDELT_API = "https://api.gdeltproject.org/api/v2/doc/doc"

HEADERS = {
    "User-Agent": "Academic research (NASCDI) - non-commercial"
}

QUERY_BLOCKS = {
    "landslide": 'landslide OR "shooting stones"',
    "snowfall": 'snowfall OR "heavy snowfall" OR "untimely snowfall" OR rainfall',
    "nh44_updates": '"Jammu Srinagar National Highway" OR "NH-44 traffic" OR "NH44 traffic"',
    "highway_closure": '"highway closed" OR "traffic suspended" OR "tunnel closed"',
    "market_arrivals": '"low arrivals" OR "arrival shortage" OR "delayed consignments"',
    "logistics": '"truck shortage" OR "freight disruption" OR "supply chain disruption"',
    "political_unrest": 'shutdown OR hartal OR "stone pelting" OR "political unrest"',
    "security": '"terrorist encounter" OR "border tension" OR "Indo Pak"',
    "covid": 'coronavirus OR covid OR lockdown',
}

CORRIDOR_CONTEXT = (
    '(Kashmir OR Srinagar OR Jammu OR Ramban OR Banihal OR Qazigund '
    'OR "Jammu Srinagar National Highway" OR "Jammu-Srinagar National Highway" '
    'OR "Srinagar Jammu National Highway" OR NH44 OR "NH-44")'
)

APPLE_MARKET_CONTEXT = (
    '((apple OR apples OR fruit OR horticulture OR "apple growers" OR "apple crop") '
    'AND (Kashmir OR Srinagar OR Sopore OR Shopian OR Baramulla OR Pulwama OR Azadpur OR Parimpora))'
)

APPLE_MARKET_BLOCKS = {"market_arrivals", "logistics"}


def context_for_block(query_block: str) -> str:
    if query_block in APPLE_MARKET_BLOCKS:
        return APPLE_MARKET_CONTEXT
    return CORRIDOR_CONTEXT


def daterange(start: datetime, end: datetime) -> Iterable[datetime]:
    day = start
    while day <= end:
        yield day
        day += timedelta(days=1)


def cache_key(query_block: str, query: str, day: datetime, maxrecords: int) -> str:
    raw = f"{query_block}|{query}|{day:%Y-%m-%d}|{maxrecords}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def cache_path(cache_dir: Path, query_block: str, query: str, day: datetime, maxrecords: int) -> Path:
    return cache_dir / "articles" / f"{cache_key(query_block, query, day, maxrecords)}.json"


def fetch(query: str, day: datetime, maxrecords: int, timeout: int) -> requests.Response:
    params = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "startdatetime": day.strftime("%Y%m%d000000"),
        "enddatetime": day.strftime("%Y%m%d235959"),
        "maxrecords": maxrecords,
        "sourcelang": "English",
    }
    return requests.get(GDELT_API, params=params, headers=HEADERS, timeout=timeout)


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


def normalize_article(article: Dict, query_block: str) -> Dict:
    title = article.get("title") or ""
    return {
        "date": article.get("seendate"),
        "title": title,
        "text": title,
        "source": article.get("domain"),
        "url": article.get("url"),
        "query_block": query_block,
        "source_country": article.get("sourcecountry"),
        "language": article.get("language"),
    }


def fetch_or_cache(
    query_block: str,
    query: str,
    day: datetime,
    cache_dir: Path,
    maxrecords: int,
    retries: int,
    timeout: int,
) -> List[Dict]:
    path = cache_path(cache_dir, query_block, query, day, maxrecords)
    if path.exists():
        return load_cached_articles(path)

    last_error = None
    for attempt in range(1, retries + 1):
        try:
            response = fetch(query, day, maxrecords=maxrecords, timeout=timeout)
            if response.status_code != 200:
                raise RuntimeError(f"HTTP {response.status_code}: {response.text[:120]}")
            payload = response.json()
            articles = payload.get("articles", [])
            save_cached_payload(path, {"articles": articles})
            return articles
        except Exception as exc:
            last_error = exc
            time.sleep((2 ** attempt) + random.random())

    fail_path = cache_dir / "failures" / f"{cache_key(query_block, query, day, maxrecords)}.txt"
    fail_path.parent.mkdir(parents=True, exist_ok=True)
    fail_path.write_text(f"{day:%Y-%m-%d} | {query_block} | {last_error}", encoding="utf-8")
    return []


def build_news_corpus(args: argparse.Namespace) -> pd.DataFrame:
    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d")
    if end < start:
        raise ValueError("--end must be on or after --start")

    selected_blocks = args.query_block or sorted(QUERY_BLOCKS)
    unknown = sorted(set(selected_blocks) - set(QUERY_BLOCKS))
    if unknown:
        raise ValueError(f"Unknown query blocks: {unknown}")

    cache_dir = Path(args.cache_dir)
    rows = []

    for query_block in selected_blocks:
        query = f"({QUERY_BLOCKS[query_block]}) AND {context_for_block(query_block)}"
        print(f"Running query block: {query_block}")

        for day in daterange(start, end):
            articles = fetch_or_cache(
                query_block=query_block,
                query=query,
                day=day,
                cache_dir=cache_dir,
                maxrecords=args.maxrecords,
                retries=args.retries,
                timeout=args.timeout,
            )
            rows.extend(normalize_article(article, query_block) for article in articles)
            time.sleep(random.uniform(args.min_sleep, args.max_sleep))

    df = pd.DataFrame(rows)
    if df.empty:
        if args.allow_empty:
            return pd.DataFrame(columns=["date", "title", "text", "source", "url", "query_block"])
        raise RuntimeError("GDELT returned no articles. Try a shorter date range, inspect failures, or broaden queries.")

    df = df.dropna(subset=["date"])
    if "url" in df:
        df["url_norm"] = df["url"].fillna("").astype(str).str.strip()
        df = df.drop_duplicates(subset=["url_norm", "query_block"], keep="first")
        df = df.drop(columns=["url_norm"])
    else:
        df = df.drop_duplicates(subset=["date", "title", "query_block"], keep="first")

    return df.sort_values(["date", "query_block", "source", "title"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a GDELT article-title corpus for news-based NASCDI.")
    parser.add_argument("--start", default="2010-01-01", help="Start date, YYYY-MM-DD.")
    parser.add_argument("--end", default="2025-12-31", help="End date, YYYY-MM-DD.")
    parser.add_argument("--out", default="data/news/raw/news_articles.csv", help="Output CSV path.")
    parser.add_argument("--cache_dir", default="data/news/cache_gdelt", help="Request cache directory.")
    parser.add_argument("--query_block", action="append", choices=sorted(QUERY_BLOCKS), help="Run one query block. Repeat for multiple blocks.")
    parser.add_argument("--maxrecords", type=int, default=50)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--min_sleep", type=float, default=0.7)
    parser.add_argument("--max_sleep", type=float, default=1.7)
    parser.add_argument("--allow_empty", action="store_true")
    args = parser.parse_args()

    df = build_news_corpus(args)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print("\nScraping complete")
    print(f"Articles: {len(df)}")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()