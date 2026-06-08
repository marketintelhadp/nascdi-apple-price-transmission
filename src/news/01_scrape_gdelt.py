"""
STEP 1 — GDELT Article Scraper for NASCDI Construction
=======================================================
Fixes applied vs original scrape_gdelt_news.py
  FIX-1: Default --start changed to 2015-02-19 (GDELT v2 reliable English coverage start)
  FIX-2: sourcelang filter REMOVED — it caused empty returns for narrow corridor queries
  FIX-3: --allow_empty flag enabled by default so early empty windows don't abort the run
  FIX-4: window_days=1 confirmed (already correct in original; Edited version had regressed this)
  FIX-5: maxrecords=250 (GDELT hard cap; Edited version had wrongly set this to 50)
  FIX-6: min_sleep/max_sleep kept at 5.5–7.5s to respect GDELT rate limits
  FIX-7: COVID block correctly included with low weight flag in lexicon (scored separately)

Run from your project root:
    python 01_scrape_gdelt.py --start 2015-02-19 --end 2025-12-31

For a quick smoke-test (single block, one month) run first to confirm output is non-empty:
    python 01_scrape_gdelt.py --start 2019-08-01 --end 2019-09-30 --query_block nh44_updates --allow_empty

Output:
    data/news/raw/news_articles.csv
    data/news/raw/news_articles_metadata.json
"""

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
    "User-Agent": "Academic research (NASCDI) - non-commercial"
}

# ── Query blocks ─────────────────────────────────────────────────────────────
# FIX-2: sourcelang:english REMOVED from all queries.
# The filter caused empty returns for narrow corridor queries, especially
# pre-2018 dates when English coverage of regional NH-44 reporting was sparse.
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
    "rainfall_flooding": (
        '("heavy rain" OR "heavy rainfall" OR "incessant rain" OR rainfall OR flood OR flooding '
        'OR cloudburst OR "flash flood" OR "road washed away") '
        'AND ("traffic suspended" OR "highway closed" OR "road closed" OR landslide OR blockage '
        'OR blocked OR stranded OR disruption) '
        'AND (Kashmir OR Jammu OR Srinagar OR Ramban OR Banihal OR Qazigund OR Udhampur '
        'OR NH44 OR "NH-44" OR "Jammu Srinagar National Highway" OR "Jammu-Srinagar National Highway")'
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
    "burhan_wani_unrest": (
        '("Burhan Wani" OR "Burhan Muzaffar Wani") '
        'AND (Kashmir OR Srinagar OR Anantnag OR Pulwama OR Shopian OR Sopore) '
        'AND (killed OR killing OR death OR unrest OR protest OR curfew OR shutdown '
        'OR hartal OR blockade OR "stone pelting" OR restrictions OR violence)'
    ),
    "article370_shutdown": (
        '("Article 370" OR "Article 35A" OR "abrogation of Article 370" OR "revocation of Article 370" '
        'OR "constitutional changes") '
        'AND (Kashmir OR Jammu OR Srinagar) '
        'AND (lockdown OR curfew OR shutdown OR restrictions OR blockade OR "communication blackout" '
        'OR "internet shutdown" OR "movement restriction" OR "security restrictions")'
    ),
    "internet_shutdown": (
        '("internet shutdown" OR "internet blockade" OR "mobile internet suspended" '
        'OR "mobile services suspended" OR "communication blackout" OR "communications blackout" '
        'OR "telecom services suspended" OR "broadband suspended") '
        'AND (Kashmir OR Jammu OR Srinagar OR Sopore OR Shopian OR Pulwama)'
    ),
    "security": (
        '("terrorist encounter" OR encounter OR "security restrictions" OR "border tension" OR "Indo Pak") '
        'AND (Kashmir OR Jammu OR Srinagar OR Sopore OR Shopian OR NH44 OR "NH-44")'
    ),
    "covid": (
        '(coronavirus OR covid OR "COVID-19" OR lockdown OR quarantine OR "movement restriction" '
        'OR "inter-state movement" OR "interstate movement") '
        'AND (Kashmir OR Jammu OR Srinagar OR Sopore OR Shopian OR Azadpur OR NH44 OR "NH-44") '
        'AND (transport OR truck OR trucks OR highway OR road OR market OR mandi OR apple OR fruit '
        'OR horticulture OR supply OR arrivals OR trade OR growers OR farmers OR movement)'
    ),
    "apple_transport": (
        '(apple OR apples OR fruit OR horticulture OR "apple growers" OR "apple farmers" '
        'OR "fruit growers" OR "apple crop" OR "apple harvest") '
        'AND (truck OR trucks OR transport OR freight OR highway OR road OR NH44 OR "NH-44" '
        'OR "Jammu Srinagar National Highway" OR "Jammu-Srinagar National Highway") '
        'AND (stranded OR stopped OR halted OR blocked OR shortage OR delayed OR disruption '
        'OR closure OR closed OR suspended OR loss OR losses)'
    ),
    "apple_market_general": (
        '(apple OR apples OR "apple growers" OR "apple farmers" OR "apple crop" OR "apple harvest" '
        'OR "fruit growers" OR "fruit mandi") '
        'AND (Kashmir OR Srinagar OR Sopore OR Shopian OR Baramulla OR Pulwama OR Azadpur OR Parimpora)'
    ),
}


def make_session() -> requests.Session:
    retry = Retry(
        total=4,
        connect=4,
        read=4,
        status=3,
        backoff_factor=2.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    session = requests.Session()
    session.headers.update(HEADERS)
    adapter = HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=4)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def window_ranges(start: datetime, end: datetime, window_days: int) -> Iterable[Tuple[datetime, datetime]]:
    current = start.replace(hour=0, minute=0, second=0, microsecond=0)
    final = end.replace(hour=23, minute=59, second=59, microsecond=0)
    while current <= final:
        window_end = min(current + timedelta(days=window_days) - timedelta(seconds=1), final)
        yield current, window_end
        current = window_end + timedelta(seconds=1)


def cache_key(query_block: str, query: str, start_dt: datetime, end_dt: datetime, maxrecords: int) -> str:
    raw = f"{query_block}|{query}|{start_dt:%Y-%m-%dT%H:%M:%S}|{end_dt:%Y-%m-%dT%H:%M:%S}|{maxrecords}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def cache_path(cache_dir: Path, query_block: str, query: str,
               start_dt: datetime, end_dt: datetime, maxrecords: int) -> Path:
    return cache_dir / "articles" / f"{cache_key(query_block, query, start_dt, end_dt, maxrecords)}.json"


def gdelt_params(query: str, start_dt: datetime, end_dt: datetime,
                 maxrecords: int, sort: str) -> Dict[str, str]:
    return {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "startdatetime": start_dt.strftime("%Y%m%d%H%M%S"),
        "enddatetime": end_dt.strftime("%Y%m%d%H%M%S"),
        "maxrecords": str(min(maxrecords, 250)),  # FIX-5: GDELT hard cap is 250
        "sort": sort,
        # FIX-2: sourcelang parameter intentionally omitted
    }


def fetch(session: requests.Session, query: str, start_dt: datetime,
          end_dt: datetime, maxrecords: int, timeout: int, sort: str) -> Dict:
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
        return json.load(f).get("articles", [])


def save_cached_payload(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    tmp.replace(path)


def normalize_article(article: Dict, query_block: str, query: str,
                      start_dt: datetime, end_dt: datetime) -> Dict:
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


def fetch_or_cache(session: requests.Session, query_block: str, query: str,
                   start_dt: datetime, end_dt: datetime, cache_dir: Path,
                   maxrecords: int, retries: int, timeout: int, sort: str) -> List[Dict]:
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
            sleep_for = min(120, (2 ** attempt) + random.random())
            print(f"  retry {attempt}/{retries}: {query_block} {start_dt:%Y-%m-%d} — {exc}")
            time.sleep(sleep_for)

    fail_dir = cache_dir / "failures"
    fail_dir.mkdir(parents=True, exist_ok=True)
    (fail_dir / f"{cache_key(query_block, query, start_dt, end_dt, maxrecords)}.txt").write_text(
        f"{start_dt:%Y-%m-%d}|{query_block}|{last_error}", encoding="utf-8"
    )
    return []


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


def selected_query_blocks(args: argparse.Namespace) -> List[str]:
    blocks = args.query_block or sorted(QUERY_BLOCKS)
    unknown = sorted(set(blocks) - set(QUERY_BLOCKS))
    if unknown:
        raise ValueError(f"Unknown query blocks: {unknown}")
    return blocks


def build_news_corpus(args: argparse.Namespace) -> pd.DataFrame:
    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d")
    if end < start:
        raise ValueError("--end must be after --start")
    if args.maxrecords > 250:
        print("GDELT ArtList caps maxrecords at 250; using 250.")
        args.maxrecords = 250

    blocks = selected_query_blocks(args)
    cache_dir = Path(args.cache_dir)
    session = make_session()
    rows = []

    for query_block in blocks:
        # FIX-2: No sourcelang filter appended
        query = QUERY_BLOCKS[query_block]
        print(f"\n{'='*60}")
        print(f"Query block: {query_block}")
        print(f"Date range: {args.start} → {args.end}")

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
            if args.verbose or n > 0:
                print(f"  {start_dt:%Y-%m-%d}: {n} articles")
            if n >= args.maxrecords:
                print(
                    f"  ⚠ WARNING: hit maxrecords={args.maxrecords} ceiling for {query_block} "
                    f"on {start_dt:%Y-%m-%d}. Reduce --window_days to 1 to avoid truncation."
                )
            rows.extend(normalize_article(a, query_block, query, start_dt, end_dt) for a in articles)
            time.sleep(random.uniform(args.min_sleep, args.max_sleep))

        print(f"Block '{query_block}' total articles (pre-dedupe): {block_count}")

    df = pd.DataFrame(rows)
    if df.empty:
        if args.allow_empty:
            print("WARNING: No articles retrieved. Check GDELT API access and query parameters.")
            return pd.DataFrame(columns=["date", "title", "text", "source", "url", "query_block"])
        raise RuntimeError(
            "GDELT returned no articles. Verify:\n"
            "  1. Start date is >= 2015-02-19\n"
            "  2. Network access to api.gdeltproject.org is available\n"
            "  3. Run with --query_block nh44_updates --start 2019-08-01 --end 2019-09-30 to test one block\n"
        )

    df = dedupe_articles(df)
    return df


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_run_metadata(out_path: Path, df: pd.DataFrame, args: argparse.Namespace) -> Path:
    meta_path = out_path.with_name(f"{out_path.stem}_metadata.json")
    metadata = {
        "script": "01_scrape_gdelt.py",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version,
        "platform": platform.platform(),
        "args": vars(args),
        "fixes_applied": [
            "FIX-1: start=2015-02-19 (GDELT v2 reliable coverage start)",
            "FIX-2: sourcelang filter removed to prevent empty returns on narrow queries",
            "FIX-3: allow_empty=True so partial windows do not abort run",
            "FIX-4: window_days=1 confirmed",
            "FIX-5: maxrecords=250 (GDELT hard cap)",
        ],
        "query_blocks": selected_query_blocks(args),
        "rows_after_dedupe": int(len(df)),
        "date_range": {
            "min": "" if df.empty else str(pd.to_datetime(df["date"], errors="coerce").min()),
            "max": "" if df.empty else str(pd.to_datetime(df["date"], errors="coerce").max()),
        },
        "rows_by_query_block": (
            df["query_block"].fillna("").value_counts().sort_index().to_dict()
            if "query_block" in df.columns and not df.empty else {}
        ),
        "output_csv": str(out_path),
        "output_sha256": file_sha256(out_path) if out_path.exists() else "",
    }
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build GDELT article corpus for NASCDI construction (all bugs fixed).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # FIX-1: Start date changed from 2010-01-01 to 2015-02-19
    parser.add_argument("--start", default="2015-02-19",
                        help="Start date YYYY-MM-DD. Do NOT set before 2015-02-19 (GDELT v2 coverage start).")
    parser.add_argument("--end", default="2025-12-31", help="End date YYYY-MM-DD.")
    parser.add_argument("--out", default="data/news/raw/news_articles.csv")
    parser.add_argument("--cache_dir", default="data/news/cache_gdelt")
    parser.add_argument("--query_block", action="append", choices=sorted(QUERY_BLOCKS),
                        help="Run one block only. Repeat to run multiple. Omit for all blocks.")
    # FIX-5: maxrecords=250 (GDELT hard cap; Edited version had wrongly set to 50)
    parser.add_argument("--maxrecords", type=int, default=250)
    # FIX-4: window_days=1 is correct and preserved
    parser.add_argument("--window_days", type=int, default=1,
                        help="Days per GDELT request. Keep at 1. Increase ONLY if zero maxrecords warnings appear.")
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=60)
    # FIX-6: Rate limit respected
    parser.add_argument("--min_sleep", type=float, default=5.5)
    parser.add_argument("--max_sleep", type=float, default=7.5)
    parser.add_argument("--sort", default="DateDesc",
                        choices=["DateDesc", "DateAsc", "HybridRel"])
    # FIX-3: allow_empty enabled to prevent aborting on sparse early-period windows
    parser.add_argument("--allow_empty", action="store_true", default=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    df = build_news_corpus(args)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    meta_path = write_run_metadata(out_path, df, args)

    print(f"\n{'='*60}")
    print("Scraping complete")
    print(f"Articles after dedupe: {len(df)}")
    print(f"Saved: {out_path}")
    print(f"Metadata: {meta_path}")
    if df.empty:
        print("\n⚠ WARNING: Output is empty. See metadata for diagnostics.")
        print("  Run smoke test: python 01_scrape_gdelt.py --query_block nh44_updates "
              "--start 2019-08-01 --end 2019-09-30")


if __name__ == "__main__":
    main()
