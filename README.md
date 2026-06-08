# NASCDI + Apple Price Transmission

This repository contains the empirical pipeline for:
- building NASCDI supply-chain disruption indexes,
- merging weekly NASCDI with mandi price series,
- estimating asymmetric price transmission models,
- producing NARDL diagnostics, tables, and plots.

## Project Structure

- `src/` production code
- `config/` lexicons
- `data/` local data and generated model inputs
- `results/` generated tables and diagnostics

## Event-Calendar NASCDI

The current stored NARDL results were built from the event-calendar pipeline:

```bash
python -m src.nascdi.build_event_nascdi
python -m src.nascdi.aggregate_nascdi_weekly
python src/data/build_weekly_prices_and_posneg.py --nascdi_path data/nascdi/nascdi_event_weekly.csv --out_dir data/model
```

## News/Lexicon NASCDI

Use this pipeline when the paper claims a news/lexicon NASCDI.

1. Build or provide a real news corpus in `data/news/raw/news_articles.csv`.
   The CSV must include `date` and either `title` or `text`. Recommended
   columns are `date,title,text,source,url,query_block`.

2. Create a historical GDELT corpus. For a full historical NASCDI, use the
   GDELT GKG BigQuery exporter. The GDELT DOC/ArtList API is only suitable for
   recent/title-level checks and should not be used as the sole historical
   source for Burhan Wani, Article 370, or early COVID windows.

```bash
python -m src.news.export_gdelt_gkg_bigquery --project YOUR_GCP_PROJECT --start 2015-02-19 --end 2025-12-31 --out_dir data/news/raw_gdelt_news
```

3. Score articles with the lexicon and build the daily news NASCDI.

```bash
python -m src.nascdi.build_nascdi_final --news_dir data/news/raw_gdelt_news --lexicon config/lexicon_news.yaml --out_dir data/nascdi_news --start 2015-02-19 --end 2025-12-31 --min_score 1.0
```

4. Aggregate the news NASCDI to weekly frequency.

```bash
python -m src.nascdi.aggregate_news_nascdi_weekly --daily data/nascdi/nascdi_daily.csv --out data/nascdi/nascdi_news_weekly.csv
```

5. Build separate weekly model files from the news NASCDI.

```bash
python src/data/build_weekly_prices_and_posneg.py --nascdi_path data/nascdi/nascdi_news_weekly.csv --out_dir data/model_news
```

Keep `data/model_news` separate from `data/model` until you are ready to
replace the event-calendar model results.
