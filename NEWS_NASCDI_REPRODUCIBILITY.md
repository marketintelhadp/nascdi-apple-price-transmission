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

Use the runner script:

```powershell
.\scripts\run_news_nascdi_pipeline.ps1 -Start "2015-01-01" -End "2025-12-31" -Dependent producer
```

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
