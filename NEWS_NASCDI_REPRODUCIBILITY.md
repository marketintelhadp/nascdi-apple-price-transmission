# News/Lexicon NASCDI Reproducibility Workflow

This workflow builds an actual news/lexicon NASCDI without overwriting the older event-calendar outputs.

## Output Folders

- Raw GDELT article-title corpus: `data/news/raw_gdelt_news/`
- News NASCDI outputs: `data/nascdi_news/`
- News-based weekly model inputs: `data/model_news/`
- News-based NARDL results: `results_news/`

The older folders `data/nascdi/`, `data/model/`, and `results/` are left intact.

## Smoke Test

Run this first from the VS Code terminal at the repository root:

```powershell
$env:PYTHONUTF8="1"
python -m src.news.scrape_gdelt_news `
  --start 2025-09-01 `
  --end 2025-09-07 `
  --query_block nh44_updates `
  --maxrecords 25 `
  --window_days 1 `
  --timeout 60 `
  --out data/news/raw_gdelt_news/news_smoke_test.csv `
  --verbose
```

The output CSV should contain article rows, not only headers. A metadata JSON is written beside it.

## Full Pipeline

Use the runner script. This uses the historical GDELT GKG BigQuery table for
the full sample, because the GDELT DOC/ArtList API is not a reliable full
historical retrieval interface for 2016/2019/2020 shock windows.

```powershell
.\scripts\run_news_nascdi_pipeline.ps1 `
  -Start "2015-02-19" `
  -End "2025-12-31" `
  -GcpProject "YOUR_GOOGLE_CLOUD_PROJECT_ID" `
  -Dependent producer
```

The runner exports GDELT GKG records related to Kashmir/NH-44/apple
supply-chain disruptions, classifies them into query blocks for NH-44 closures,
landslides, snowfall, rainfall/flooding, apple transport, market arrivals,
Burhan Wani unrest, Article 370 restrictions, internet shutdowns, COVID-19,
security restrictions, and political unrest. It then audits the corpus against
the major-event windows in `config/gdelt_event_windows.csv`.

If the audit reports a missing major event, inspect and broaden that event's
query before building the index. Use `-AllowMissingMajorEvents` only for
diagnostic runs, not for the final replication build.

Before the first BigQuery run, install the optional client and authenticate:

```powershell
python -m pip install google-cloud-bigquery
gcloud auth application-default login
```

If you do not have the Google Cloud CLI, authenticate through your institution's
preferred BigQuery workflow and make sure Application Default Credentials are
available to Python.

Use `-SkipScrape` when the raw GDELT CSVs are already present and you only want to rebuild the index, model files, and NARDL results:

```powershell
.\scripts\run_news_nascdi_pipeline.ps1 -Start "2015-01-01" -End "2025-12-31" -Dependent producer -SkipScrape
```

## Manual Commands

Build NASCDI from the raw news corpus:

```powershell
python -m src.nascdi.build_nascdi_final `
  --news_dir data/news/raw_gdelt_news `
  --lexicon config/lexicon_news.yaml `
  --out_dir data/nascdi_news `
  --start 2015-01-01 `
  --end 2025-12-31 `
  --min_score 1 `
  --clip_raw 30 `
  --normalize z_to_100_10 `
  --sensitivity_lexicon config/lexicon_sensitivity.yaml
```

Audit major-event and annual query-block coverage:

```powershell
python -m src.news.audit_gdelt_event_coverage `
  --news_dir data/news/raw_gdelt_news `
  --event_windows config/gdelt_event_windows.csv `
  --out_dir results_news/coverage_audit `
  --fail_on_missing
```

Estimate BigQuery bytes before exporting:

```powershell
python -m src.news.export_gdelt_gkg_bigquery `
  --project YOUR_GOOGLE_CLOUD_PROJECT_ID `
  --start 2015-02-19 `
  --end 2025-12-31 `
  --out_dir data/news/raw_gdelt_news `
  --dry_run
```

Build weekly model inputs:

```powershell
python -m src.data.build_weekly_prices_and_posneg `
  --price_path data/clean/prices_long.csv `
  --nascdi_path data/nascdi_news/nascdi_news_weekly.csv `
  --out_dir data/model_news `
  --min_weeks 60
```

Run NARDL. Use `producer` if the manuscript models producer/farmgate prices as the dependent variable:

```powershell
python -m src.models.run_nardl_batch_log `
  --in_dir data/model_news `
  --out_dir results_news/nardl `
  --dependent producer
```

Validate the full pipeline:

```powershell
python -m src.nascdi.validate_news_pipeline `
  --news_dir data/news/raw_gdelt_news `
  --nascdi_dir data/nascdi_news `
  --model_dir data/model_news `
  --results_dir results_news/nardl `
  --start 2015-01-01 `
  --end 2025-12-31 `
  --out results_news/news_pipeline_validation.json
```

## Journal Rule

Do not claim that the old `data/nascdi/nascdi_event_weekly.csv` results are news/lexicon results. For a defensible paper, cite the outputs in `data/nascdi_news/`, `data/model_news/`, and `results_news/`, plus the metadata JSON files and validation report.

If GDELT returns insufficient pre-2015 article coverage, either restrict the news/lexicon sample to the verified coverage period or add a documented external archive such as Factiva/ProQuest/manual newspaper records for 2010-2014. Do not fill missing early news years with synthetic or event-calendar NASCDI and label it news-based.
