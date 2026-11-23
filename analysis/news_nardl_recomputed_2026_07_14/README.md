# Recomputed Targeted-GDELT News NASCDI NARDL Package

This package is separate from the existing event-calendar results. It uses:

- Weekly targeted-GDELT news NASCDI: `E:\NASCDI APPLE PRICE TRANSMISSION PAPER\nascdi-apple-price-transmission\data\nascdi_news\nascdi_news_weekly.csv`
- Weekly price data: `E:\NASCDI APPLE PRICE TRANSMISSION PAPER\nascdi-apple-price-transmission\data\clean\prices_long.csv`
- Dependent variable: log terminal-market price.
- Price regressor: log producer-market price.
- NARDL/ECM lag order: four lags for the dependent variable and all regressors.
- Inference: Newey-West HAC standard errors with a four-week bandwidth.

## Contents

- `data/model_news`: rebuilt price/NASCDI data for the eight estimated series, plus reproducibility metadata.
- `results_news/nardl`: batch-estimation output and its reproducibility metadata.
- `results_news/nardl/tables`: HAC parameter estimates, model summary, diagnostics, and simulated dynamic responses.
- `results_news/nardl/plots`: five 300-dpi figures.
- `MODEL_EQUATIONS.md`: the NASCDI and NARDL equations used for this run.
- `create_news_nardl_report.py`: reproducible report generator.
- `install_into_vscode_and_push.ps1`: copies the package to the VS Code project,
  creates a dedicated Git branch, commits only the new analysis, and pushes it.

## Install and publish

Run the installer from PowerShell under your normal Windows account:

```powershell
& "C:\Users\masro\Documents\Codex\2026-05-20\files-mentioned-by-the-user-nascdi\news_nardl_recomputed\install_into_vscode_and_push.ps1"
```

It places the package under `analysis/news_nardl_recomputed_2026_07_14` in the
repository. It includes the exact daily and weekly NASCDI files, model inputs,
tables, plots, and equations, but deliberately excludes raw GDELT downloads and
the 8.2 GB `news_scored.csv` audit file. The script then pushes the branch
`codex/news-nascdi-recomputation-2026-07-14` to `origin`.

## Important interpretation limits

The NASCDI is a targeted GDELT GKG metadata-derived disruption-window index, not an all-articles full-text corpus. Describe it precisely in the manuscript. The estimates show stable negative and HAC-significant error-correction terms in all eight models, but no long-run or short-run NASCDI asymmetry rejection at the 5% level. Do not claim statistically established NASCDI asymmetry from this run.

Several series reject homoskedasticity and normality in residual diagnostics. HAC inference addresses heteroskedasticity/autocorrelation in standard errors, but this should be reported and accompanied by robustness checks rather than hidden.
