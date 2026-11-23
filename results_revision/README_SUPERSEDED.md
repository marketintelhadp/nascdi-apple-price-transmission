# Earlier revision round (before the corpus rebuild)

The files in this folder were produced from the first, incomplete GDELT extract. Everything that depends on the NASCDI is **superseded** by
`results_rebuild_v2/`: `main_*.csv`, `log_*.txt` (including the regime-standardised, post-2018 and pre-2018 checks, which the revised paper drops),
`index_coverage_by_year.csv`, `break_*.csv`, `dnascdi_*.csv`, `partial_sum_summary.csv`, `identification_diagnostics.csv`, `cusum_revised.csv`,
`stability_and_inference.csv`, `dynamic_responses_representative_shocks.csv` and the figures.

Outputs that use **prices only** are unaffected by the rebuild and remain valid: `toda_yamamoto.csv`, the price rows of `unit_root_tests.csv`,
and `audit/power_cointegration.csv` with the spread test (Appendix Table A12; not re-run on the rebuilt sample sizes).
