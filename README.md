# NASCDI + Apple Price Transmission (Parimpora → Azadpur)

This repository contains the complete research pipeline for:
- building a News-based Agricultural Supply Chain Disruption Index (NASCDI) from digital news (2010–2025),
- integrating NASCDI with daily mandi prices and arrivals (2015–2025),
- estimating asymmetric price transmission (NARDL),
- forecasting using multi-channel LSTM and hybrid LSTM–NARDL,
- producing policy-ready early warning outputs.

## Project Structure
- `src/` production code
- `notebooks/` analysis notebooks
- `config/` lexicon and project settings
- `data/` local data (not tracked in git)
- `outputs/` generated tables/figures/models (not tracked in git)

## Build NASCDI
1) Place news CSV files in: `data/news/raw/`
2) Ensure CSV contains at least: `date`, and `text` or `title`
3) Run:

```bash
python -m src.nascdi.build_nascdi --news_dir data/news/raw --lexicon config/lexicon.yaml --out_dir data/nascdi --min_score 1.0
