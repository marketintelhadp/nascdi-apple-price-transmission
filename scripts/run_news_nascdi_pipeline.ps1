param(
    [string]$Start = "2015-02-19",
    [string]$End = "2025-12-31",
    [string]$GcpProject = "",
    [ValidateSet("producer", "terminal")]
    [string]$Dependent = "producer",
    [switch]$SkipScrape,
    [switch]$AllowMissingMajorEvents
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $repo
$env:PYTHONUTF8 = "1"

if (-not $SkipScrape) {
    if ([string]::IsNullOrWhiteSpace($GcpProject)) {
        throw "Provide -GcpProject <your-google-cloud-project-id> for the historical GDELT GKG export."
    }

    python -m src.news.export_gdelt_gkg_bigquery `
        --project $GcpProject `
        --start $Start `
        --end $End `
        --out_dir data/news/raw_gdelt_news
}

if ($AllowMissingMajorEvents) {
    python -m src.news.audit_gdelt_event_coverage `
        --news_dir data/news/raw_gdelt_news `
        --event_windows config/gdelt_event_windows.csv `
        --out_dir results_news/coverage_audit
}
else {
    python -m src.news.audit_gdelt_event_coverage `
        --news_dir data/news/raw_gdelt_news `
        --event_windows config/gdelt_event_windows.csv `
        --out_dir results_news/coverage_audit `
        --fail_on_missing
}

python -m src.nascdi.build_nascdi_final `
    --news_dir data/news/raw_gdelt_news `
    --lexicon config/lexicon_news.yaml `
    --out_dir data/nascdi_news `
    --start $Start `
    --end $End `
    --min_score 1 `
    --clip_raw 30 `
    --normalize z_to_100_10 `
    --sensitivity_lexicon config/lexicon_sensitivity.yaml

python -m src.data.build_weekly_prices_and_posneg `
    --price_path data/clean/prices_long.csv `
    --nascdi_path data/nascdi_news/nascdi_news_weekly.csv `
    --out_dir data/model_news `
    --min_weeks 60

python -m src.models.run_nardl_batch_log `
    --in_dir data/model_news `
    --out_dir results_news/nardl `
    --dependent $Dependent

python -m src.nascdi.validate_news_pipeline `
    --news_dir data/news/raw_gdelt_news `
    --nascdi_dir data/nascdi_news `
    --model_dir data/model_news `
    --results_dir results_news/nardl `
    --start $Start `
    --end $End `
    --out results_news/news_pipeline_validation.json
