import argparse
import hashlib
import json
import re
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

try:
    from google.cloud import bigquery
except ImportError as exc:
    raise SystemExit(
        "google-cloud-bigquery is required. Install it with: "
        "python -m pip install google-cloud-bigquery"
    ) from exc


GKG_TABLE = "gdelt-bq.gdeltv2.gkg_partitioned"

CONTEXT_REGEX = (
    r"kashmir|jammu|srinagar|shopian|sopore|pulwama|baramulla|anantnag|ramban|"
    r"banihal|qazigund|udhampur|azadpur|parimpora|jammu[-_ ]srinagar|nh[-_ ]?44|nh[-_ ]?1a"
)

SHOCK_REGEX = (
    r"landslide|mudslide|rockfall|shooting[-_ ]stones|snowfall|snowstorm|avalanche|"
    r"rainfall|heavy[-_ ]rain|flood|cloudburst|road[-_ ]closed|highway[-_ ]closed|"
    r"traffic[-_ ]suspended|traffic[-_ ]halted|stranded[-_ ]trucks|truck[-_ ]shortage|"
    r"apple[-_ ]truck|fruit[-_ ]truck|low[-_ ]arrivals|arrival[-_ ]shortage|"
    r"market[-_ ]glut|distress[-_ ]sale|shutdown|hartal|bandh|curfew|blockade|"
    r"internet[-_ ]shutdown|communication[-_ ]blackout|article[-_ ]370|article[-_ ]35a|"
    r"burhan[-_ ]wani|covid|coronavirus|lockdown|movement[-_ ]restriction"
)

THEME_REGEX = (
    r"NATURAL_DISASTER|ENV_.*DISASTER|CRISISLEX_.*|PROTEST|UNREST|STRIKE|"
    r"BLOCKADE|CURFEW|SHORTAGE|TRANSPORT|ROAD|FLOOD|AVALANCHE|LANDSLIDE|"
    r"SNOW|RAIN|VIRUS|EPIDEMIC|PANDEMIC|INTERNET"
)

CATEGORY_PATTERNS = {
    "burhan_wani_unrest": r"burhan wani|burhan muzaffar wani",
    "article370_shutdown": r"article 370|article 35a|abrogation|revocation",
    "internet_shutdown": r"internet shutdown|internet blockade|mobile internet|communication blackout|communications blackout",
    "covid": r"covid|covid-19|coronavirus|pandemic|quarantine",
    "landslide": r"landslide|landslides|mudslide|rockfall|shooting stones",
    "snowfall": r"snowfall|heavy snowfall|snowstorm|avalanche",
    "rainfall_flooding": r"heavy rain|heavy rainfall|incessant rain|flood|flooding|cloudburst|flash flood",
    "highway_closure": r"highway closed|road closed|traffic suspended|traffic halted|road blocked|highway blocked",
    "apple_transport": r"apple truck|fruit truck|stranded truck|truck shortage|delayed consignment",
    "market_arrivals": r"low arrivals|arrival shortage|restricted arrivals|market glut|distress sale",
    "political_unrest": r"shutdown|hartal|bandh|curfew|blockade|political unrest|stone pelting",
}


QUERY = f"""
SELECT
  DATE,
  SourceCollectionIdentifier,
  SourceCommonName,
  DocumentIdentifier,
  V2Themes,
  V2Locations,
  V2Persons,
  V2Organizations,
  V2Tone
FROM `{GKG_TABLE}`
WHERE _PARTITIONTIME >= TIMESTAMP(@start_date)
  AND _PARTITIONTIME < TIMESTAMP(@end_date_exclusive)
  AND (
    REGEXP_CONTAINS(LOWER(IFNULL(V2Locations, '')), @context_regex)
    OR REGEXP_CONTAINS(LOWER(IFNULL(DocumentIdentifier, '')), @context_regex)
    OR REGEXP_CONTAINS(LOWER(IFNULL(V2Persons, '')), r'burhan wani')
  )
  AND (
    REGEXP_CONTAINS(LOWER(IFNULL(DocumentIdentifier, '')), @shock_regex)
    OR REGEXP_CONTAINS(IFNULL(V2Themes, ''), @theme_regex)
    OR REGEXP_CONTAINS(LOWER(IFNULL(V2Persons, '')), r'burhan wani')
  )
"""


def month_ranges(start: date, end: date):
    current = start.replace(day=1)
    while current <= end:
        if current.month == 12:
            next_month = current.replace(year=current.year + 1, month=1)
        else:
            next_month = current.replace(month=current.month + 1)
        chunk_start = max(current, start)
        chunk_end = min(next_month - timedelta(days=1), end)
        yield chunk_start, chunk_end
        current = next_month


def url_slug(url: str) -> str:
    value = re.sub(r"[?#].*$", "", str(url))
    value = value.rstrip("/").rsplit("/", 1)[-1]
    value = re.sub(r"[-_]+", " ", value)
    value = re.sub(r"\.(html?|aspx?|php)$", "", value, flags=re.I)
    return re.sub(r"\s+", " ", value).strip()


def clean_gkg_field(value: str) -> str:
    value = "" if pd.isna(value) else str(value)
    value = re.sub(r"[#;,]", " ", value)
    value = re.sub(r"\d+(?:\.\d+)?", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def classify_query_blocks(text: str) -> str:
    blocks = [name for name, pattern in CATEGORY_PATTERNS.items() if re.search(pattern, text, flags=re.I)]
    return "|".join(blocks) if blocks else "historical_gkg_context"


def normalize_results(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "title",
                "text",
                "source",
                "url",
                "query_block",
                "source_country",
                "language",
                "gdelt_source",
                "gdelt_tone",
            ]
        )

    df = df.copy()
    df["date"] = pd.to_datetime(df["DATE"].astype(str), format="%Y%m%d%H%M%S", errors="coerce")
    df["url"] = df["DocumentIdentifier"].fillna("").astype(str)
    df["title"] = df["url"].map(url_slug)
    metadata_text = (
        df["title"].fillna("")
        + " "
        + df["V2Themes"].map(clean_gkg_field)
        + " "
        + df["V2Locations"].map(clean_gkg_field)
        + " "
        + df["V2Persons"].map(clean_gkg_field)
        + " "
        + df["V2Organizations"].map(clean_gkg_field)
    ).str.lower()
    df["text"] = metadata_text
    df["query_block"] = metadata_text.map(classify_query_blocks)
    df["source"] = df["SourceCommonName"].fillna("").astype(str)
    df["source_country"] = ""
    df["language"] = ""
    df["gdelt_source"] = "GKG BigQuery metadata"
    df["gdelt_tone"] = df["V2Tone"].fillna("").astype(str)

    out_cols = [
        "date",
        "title",
        "text",
        "source",
        "url",
        "query_block",
        "source_country",
        "language",
        "gdelt_source",
        "gdelt_tone",
    ]
    out = df[out_cols].dropna(subset=["date"]).copy()
    out["url_norm"] = out["url"].str.strip()
    out = out.sort_values("date").drop_duplicates("url_norm", keep="first").drop(columns="url_norm")
    return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def query_month(client, project: str, start: date, end: date, dry_run: bool):
    end_exclusive = end + timedelta(days=1)
    config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("start_date", "STRING", start.isoformat()),
            bigquery.ScalarQueryParameter("end_date_exclusive", "STRING", end_exclusive.isoformat()),
            bigquery.ScalarQueryParameter("context_regex", "STRING", CONTEXT_REGEX),
            bigquery.ScalarQueryParameter("shock_regex", "STRING", SHOCK_REGEX),
            bigquery.ScalarQueryParameter("theme_regex", "STRING", THEME_REGEX),
        ],
        dry_run=dry_run,
        use_query_cache=not dry_run,
    )
    job = client.query(QUERY, project=project, job_config=config)
    if dry_run:
        return None, int(job.total_bytes_processed or 0)
    return job.result().to_dataframe(create_bqstorage_client=False), int(job.total_bytes_processed or 0)


def main():
    parser = argparse.ArgumentParser(
        description="Export historical Kashmir/NH-44 disruption records from GDELT GKG BigQuery.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--project", required=True, help="Your Google Cloud billing/project ID.")
    parser.add_argument("--start", default="2015-02-19")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--out_dir", default="data/news/raw_gdelt_gkg")
    parser.add_argument("--dry_run", action="store_true", help="Estimate query bytes without exporting data.")
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    if end < start:
        raise ValueError("--end must be on or after --start")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    client = bigquery.Client(project=args.project)
    manifest = []

    for chunk_start, chunk_end in month_ranges(start, end):
        print(f"Querying GKG: {chunk_start} to {chunk_end}")
        raw, bytes_processed = query_month(client, args.project, chunk_start, chunk_end, args.dry_run)
        if args.dry_run:
            print(f"  Estimated bytes: {bytes_processed:,}")
            manifest.append(
                {
                    "start": chunk_start.isoformat(),
                    "end": chunk_end.isoformat(),
                    "estimated_bytes": bytes_processed,
                }
            )
            continue

        normalized = normalize_results(raw)
        path = out_dir / f"gdelt_gkg_{chunk_start:%Y_%m}.csv"
        normalized.to_csv(path, index=False)
        print(f"  Saved {len(normalized):,} records: {path}")
        manifest.append(
            {
                "start": chunk_start.isoformat(),
                "end": chunk_end.isoformat(),
                "bytes_processed": bytes_processed,
                "rows": int(len(normalized)),
                "path": str(path),
                "sha256": sha256(path),
            }
        )

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "src/news/export_gdelt_gkg_bigquery.py",
        "table": GKG_TABLE,
        "project": args.project,
        "start": args.start,
        "end": args.end,
        "dry_run": args.dry_run,
        "context_regex": CONTEXT_REGEX,
        "shock_regex": SHOCK_REGEX,
        "theme_regex": THEME_REGEX,
        "chunks": manifest,
        "method_note": (
            "Historical GDELT GKG contains metadata, not guaranteed article full text. "
            "The title is reconstructed from the article URL slug and text combines GKG metadata."
        ),
    }
    metadata_path = out_dir / "gdelt_gkg_export_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved metadata: {metadata_path}")


if __name__ == "__main__":
    main()
