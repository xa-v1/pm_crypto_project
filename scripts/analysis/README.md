# scripts/analysis/

Source of every figure in `Presentation-OQG-Banquet.pdf` (repo root). Each script reads from `data/` and writes to `data/cleaned/` and `figures/`.

## Headline diagnostics (deck slides 10, 12, 14)

| Script | Output | Slide |
|---|---|---|
| `lead_lag.py` | `data/cleaned/pm_spot_lead_lag.json`, `figures/lead_lag.png` | 10 |
| `variance_ratio.py` | `data/cleaned/variance_ratio.json` | 12 (data) |
| `make_figures/make_variance_ratio_panel.py` | `figures/variance_ratio.png` | 12 |
| `logReg/incremental_information.py` | `data/cleaned/incremental_summary.json`, `data/cleaned/incremental_oos.csv`, `figures/logReg/incremental_information.png` | 14 |

## Data section (deck slides 4–8)

| Script | Output | Slide |
|---|---|---|
| `make_figures/make_data_joint_panel.py` | `figures/eda/data_joint_panel.png` | 4 |
| `eda/preprocessing.py` | contract feature CSVs + `figures/eda/probability_convergence.png` + `figures/eda/calibration_panel.png` | 5 + appendix |
| `calibration/make_hourly_accuracy.py` | `figures/eda/hourly_accuracy.png` | 6 |
| `calibration/make_accuracy_heatmap.py` | `figures/eda/accuracy_heatmap.png` | 7 |
| `make_figures/make_diagrams.py` | `figures/eda/methods_diagram.png` | 8 |

## Appendix

| Script | Output |
|---|---|
| `eda/preprocessing.py` | `figures/eda/calibration_panel.png` (Calibration) |
| `logReg/make_conviction_spread_accuracy.py` | `figures/eda/conviction_spread_accuracy.png` (Market Disagreement Signal) |
| `logReg/make_lr_appendix_figures.py` | `figures/logReg/feature_importance.png` (LR Feature Importance) |

## Supplementary research

Directly extends the deck's stated limitation and next-step on slide 15 — characterizes what we'd build on next but did not present.

| Script | Output |
|---|---|
| `logReg/make_volatility_forecast.py` | `figures/logReg/volatility_forecast.png` + `data/cleaned/volatility_forecast.json` — HAR-RV vs GARCH(1,1) vs Kalshi conviction spread for next-15-min realized vol |
| `stress_events.py` | `figures/stress_events_lead_lag.png` + `data/cleaned/stress_events.json` — lead-lag correlogram restricted to top vs bottom realized-vol decile of contracts |

## Data utilities

- `eda/pull_coinbase_spot.py` — fetches Coinbase BTC-USD and ETH-USD 1-minute OHLCV bars.

## Methodological notes

- Cluster bootstrap by contract (Cameron, Gelbach & Miller 2008) for within-contract Kalshi statistics.
- Stationary block bootstrap (Politis & Romano 1994) for continuous spot CIs in the VR test.
- 5-fold time-series cross-validation (train past, test future) in the incremental info test.
- DeLong paired test (1988) for AUC comparisons on shared OOS data.
- Spot features (8) and Kalshi features (9) are each venue's natural set; the asymmetry favors Kalshi.
