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


---

## Revision for Scientific Reports (November 2026)

The revised manuscript ("News-Based Corridor Disruption and Producer-Terminal Price
Transmission in Kashmir's Apple Sector: A NASCDI-NARDL Analysis") is produced by the code in
`src/revision/` and the outputs in `results_revision/`. **The revised paper does not use the
older estimation outputs** in `results/`, `results_vi/`, `data/model/` or the scripts in
`src/models/` (see "Superseded material" below).

### What changed relative to the submitted version

1. Lags and differences are taken over **calendar weeks** on a regular weekly grid; an observation
   enters only if every lag it needs is observed in consecutive weeks (the submitted code
   differenced across gaps of up to 89 weeks).
2. NASCDI- is the **Shin et al. (2014) negative partial sum** (non-positive), so that
   beta+ = beta- is the symmetry null (the submitted code stored absolute decreases).
3. Lag orders are selected per equation by AIC subject to a Breusch-Godfrey check.
4. Cointegration uses the **bounds F- and t-tests with finite-sample critical values simulated per
   equation**, checked with a regressor-conditional wild bootstrap.
5. Symmetry (long-run, short-run, joint) uses HAC Wald statistics and **restricted wild-bootstrap
   p-values**.
6. The vulnerability index (VI) and the CDII protocol are no longer part of the paper.

### Requirements before running

The revision scripts read `data/model_news/` (weekly chain files) and
`data/nascdi_news/nascdi_news_weekly.csv`. Rebuild them with the "News/Lexicon NASCDI" steps above, or copy the identical, already-built inputs from
`analysis/news_nardl_recomputed_2026_07_14/data/` (`model_news/` and `nascdi_news/`) into `data/`.
Run all commands from the repository root.

### Main analysis (`src/revision/`)

| Script | Output in `results_revision/` | Paper |
|---|---|---|
| `run_main.py [label]` | `main_base.csv` | Tables 6, 8; Figs 7, 8, 10 |
| `run_robust.py` | `main_regime_z.csv`, `main_post2018.csv`, `main_pre2018.csv`, `main_reverse.csv`, `main_covid.csv` | Table 10; Appendix Table A5 |
| `run_index_diagnostics.py` | `index_coverage_by_year.csv`, `break_ic.csv`, `break_seq.csv`, `dnascdi_*.csv`, `partial_sum_summary.csv`, `identification_diagnostics.csv`, `fig_index_coverage.png` | Tables 4, 7; Fig 4; Appendix Tables A3, A8 |
| `run_unitroot.py` | `unit_root_tests.csv`, `unit_root_summary.csv` | Table 2; Appendix Table A2 |
| `run_stability.py` | `stability_and_inference.csv` | Appendix Table A4 |
| `run_cusum.py` | `cusum_revised.csv` | Table 9; Appendix Table A7 |
| `run_toda_yamamoto.py` | `toda_yamamoto.csv` | Appendix Table A1 |
| `run_multipliers.py` | `dynamic_responses_representative_shocks.csv` | Fig 11; Appendix Table A6 |
| `make_figures.py` | `figs/` | Figs 7-11 (file names `fig6_ect_ci`, `fig7_bounds_F`, `fig_partial_sums`, `fig8_symmetry_p`, `fig10_multipliers` are kept from earlier drafts) |
| `make_cusum_figs.py` | `figs/fig12_cusum.png`, `figs/fig13_cusumsq.png` | Figs 12-13 |
| `make_direction_fig.py` | `figs/fig14_direction.png` | Fig 14 |

Core functions are in `rev_lib.py` (design, partial sums, Wald and wild-bootstrap tests,
simulated bounds) and `rev_run.py` (per-chain estimation).

### Additional checks (`src/revision/audit/`, outputs in `results_revision/audit/`)

| Script | Output | Paper |
|---|---|---|
| `a2_boot.py 999 -` and `a2_boot.py 999 - rev` | `bounds_bootstrap_producer.csv`, `bounds_bootstrap_reverse.csv` | Section 5.7, Fig 14; Appendix Table A9 |
| `a9_stale.py` | `stale_price_filter.csv` | Section 5.7; Appendix Table A10 |
| `a3_nascdi.py`, `a10_revnas.py`, `a11_cumulative.py`, `a4_power.py` | `nascdi_exclusion_and_spread.csv`, `nascdi_terminal_equation.csv`, `nascdi_cumulative_effect.csv`, `nascdi_detectable_effect.csv`, `nascdi_pooled.csv` | Section 5.7; Appendix Table A11 |
| `a3_nascdi.py` (spread test), `a5_power_cal.py` | `nascdi_exclusion_and_spread.csv`, `power_cointegration.csv` | Appendix Table A12 |
| `a1_data.py` | `log_data_checks.txt` | week-grid and gap checks |

`a2_boot.py` takes `B` (bootstrap replications), a chain filter (`-` for all) and an optional `rev`
flag for the terminal-price-dependent equation. The `log_*.txt` files are console output of the
runs that produced the stored csv files.

Not in this repository: the equal-weights re-scoring of the corpus (Appendix Table A13) was run
outside the repository and its scripts and outputs still need to be added.

### Superseded material (kept for provenance only)

- `src/models/run_nardl_batch.py`, `src/models/run_nardl_batch_log.py` and the earlier test and
  diagnostic scripts in `src/models/nardl/`: the submitted-version estimation (row-based lags; the
  negative partial sum stored as absolute decreases; ordinary F p-values in place of bounds
  critical values).
- `results/`, `data/model/` (event-calendar pipeline) and `results_news/` (not tracked): outputs of that
  estimation.
- `results_vi/` and `structural_sensitivity_vulnerability_summary.py`: vulnerability-index outputs
  from the submitted version.

Do not use these to reproduce the revised paper.
