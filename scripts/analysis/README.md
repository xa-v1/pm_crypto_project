# scripts/analysis/

Source of every figure in the presentation. Each script reads from `data/` and writes to `data/cleaned/` and `figures/`.

## Headline diagnostics (deck slides 10–13)

| Script | Output |
|---|---|
| `lead_lag.py` | `data/cleaned/pm_spot_lead_lag.json`, `figures/lead_lag.png` |
| `variance_ratio.py` | `data/cleaned/variance_ratio.json` |
| `make_figures/make_variance_ratio_panel.py` | `figures/variance_ratio.png` (slide 12) |
| `logReg/incremental_information.py` | `data/cleaned/incremental_summary.json`, `data/cleaned/incremental_oos.csv`, `figures/logReg/incremental_information.png` |

## Data section (deck slides 6–9)

| Script | Output |
|---|---|
| `eda/preprocessing.py` | contract feature CSVs + `figures/eda/probability_convergence.png` (slide 7) + `figures/eda/calibration_panel.png` (appendix slide 19) |
| `make_figures/make_data_joint_panel.py` | `figures/eda/data_joint_panel.png` (slide 6) |
| `calibration/make_hourly_accuracy.py` | `figures/eda/hourly_accuracy.png` (slide 8) |
| `calibration/make_accuracy_heatmap.py` | `figures/eda/accuracy_heatmap.png` (slide 9) |
| `make_figures/make_diagrams.py` | `figures/eda/methods_diagram.png` (slide 10) |

## Appendix (deck slides 19–22 + follow-ups)

| Script | Output |
|---|---|
| `eda/preprocessing.py` | `figures/eda/calibration_panel.png` (slide 19) |
| `logReg/make_conviction_spread_accuracy.py` | `figures/eda/conviction_spread_accuracy.png` (slide 20) |
| `logReg/make_lr_appendix_figures.py` | `figures/logReg/feature_importance.png` + `figures/logReg/decision_boundary.png` (slides 21–22) |
| `logReg/make_volatility_forecast.py` | `figures/logReg/volatility_forecast.png` + `data/cleaned/volatility_forecast.json` — second-moment channel: HAR-RV vs GARCH(1,1) vs Kalshi conviction spread for next-15-min realized vol |
| `stress_events.py` | `figures/stress_events_lead_lag.png` + `data/cleaned/stress_events.json` — lead-lag correlogram restricted to top vs bottom realized-vol decile of contracts |

## Data utilities

- `eda/pull_coinbase_spot.py` — fetches Coinbase BTC-USD and ETH-USD 1-minute OHLCV bars.

## Methodological notes

- Cluster bootstrap by contract (Cameron, Gelbach & Miller 2008) for within-contract Kalshi statistics.
- Stationary block bootstrap (Politis & Romano 1994) for continuous spot CIs in the VR test.
- 5-fold time-series cross-validation (train past, test future) in the incremental info test.
- DeLong paired test (1988) for AUC comparisons on shared OOS data.
- Spot features (8) and Kalshi features (9) are each venue's natural set; the asymmetry favors Kalshi.
