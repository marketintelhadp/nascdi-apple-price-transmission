param(
    [string]$Start = "2015-01-01",
    [string]$End = "2025-12-31",
    [ValidateSet("producer", "terminal")]
    [string]$Dependent = "producer",
    [switch]$SkipScrape
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $repo
$env:PYTHONUTF8 = "1"

$blocks = @(
    "landslide",
    "snowfall",
    "nh44_updates",
    "highway_closure",
    "market_arrivals",
    "logistics",
    "political_unrest",
    "security",
    "covid",
    "apple_market_general"
)

if (-not $SkipScrape) {
    foreach ($block in $blocks) {
        $out = "data/news/raw_gdelt_news/news_${block}_${Start}_${End}.csv"
        python -m src.news.scrape_gdelt_news `
            --start $Start `
            --end $End `
            --query_block $block `
            --maxrecords 250 `
            --window_days 1 `
            --timeout 60 `
            --min_sleep 5.5 `
            --max_sleep 7.5 `
            --out $out `
            --verbose
    }
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
