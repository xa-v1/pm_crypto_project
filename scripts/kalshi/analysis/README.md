# Analysis pipeline

Code that produces the headline diagnostics, the calibration figures, and the appendix analyses. All scripts read from `data/` and write to `data/cleaned/` and `figures/<topic>/` using paths relative to the repo root.

---

## Headline diagnostics (slides 10–12)

These are the three tests reported in the presentation.

| Subfolder | Script | Test | Output |
|---|---|---|---|
| `lagCorr/` | `pm_spot_lead_lag.py` | Pearson cross-correlogram of ΔP(UP)_t against r_{t+k}, k ∈ [−10, +10] | `data/cleaned/pm_spot_lead_lag.json`, `figures/lagCorr/lead_lag.png` |
| `varRatio/` | `variance_ratio.py` | Lo & MacKinlay (1988) VR(q) test on P(UP) vs spot returns | `data/cleaned/variance_ratio.json`, `figures/varRatio/variance_ratio.png` |
| `logReg/` | `incremental_information.py` | Spot-only vs Kalshi-only vs Combined logistic regression with DeLong AUC test | `data/cleaned/incremental_summary.json`, `data/cleaned/incremental_oos.csv`, `figures/logReg/incremental_information.png` |

### Run from the repo root

```bash
python -m scripts.kalshi.analysis.lagCorr.pm_spot_lead_lag
python -m scripts.kalshi.analysis.varRatio.variance_ratio
python -m scripts.kalshi.analysis.logReg.incremental_information
```

---

## Calibration analyses (Xavi's section, slides 6–8)

`calibration/` — calibration, accuracy, and intraday seasonality on the Kalshi BTC panel.

| Script | Output |
|---|---|
| `btc_accuracy.py` | `figures/eda/btc_accuracy_timeseries.png` |
| `btc_heatmap.py` | `figures/eda/accuracy_heatmap.png` |
| `btc_hourly_analysis.py` | `figures/eda/hourly_accuracy.png` |
| `btc_entry_strategy.py` | `figures/eda/btc_entry_table.png` |
| `plot_brier_score.py` | `figures/eda/btc_brier_score_timeseries.png` |

Each script can be run independently:

```bash
python scripts/kalshi/analysis/calibration/btc_accuracy.py
```

---

## Exploratory data analysis

`eda/`

| File | Role |
|---|---|
| `data_exp.ipynb` | Jupyter notebook with cross-asset EDA, regime stability checks, disagreement signal |
| `preprocessing.py` | Feature-engineering helpers reused by `logReg/logistic_regression.py` and `xgboost_model.py` |
| `pull_coinbase_spot.py` | Fetches Coinbase BTC-USD 1-minute OHLCV bars over the project window |

---

## Appendix analyses (kept, not in the headline)

| Script | Test |
|---|---|
| `lagCorr/cross_asset_lead_lag.py` | Does BTC P(UP) lead ETH spot (or vice versa)? |
| `lagCorr/volatility_leadership.py` | Does \|ΔP(UP)\| lead \|spot return\|? (second-moment version of the lead-lag test) |
| `lagCorr/lead_lag.py` | Older BTC vs ETH P(UP) cross-correlogram (kept for reproducibility of earlier scope) |
| `logReg/logistic_regression.py` | Earlier BTC+ETH logistic regression baseline + extended models |
| `logReg/xgboost_model.py` | XGBoost benchmark for the earlier scope |

---

## Figure generators

`make_figures/` — pure plotting code that builds presentation figures from cached JSON/CSV outputs.

| Script | Produces |
|---|---|
| `make_diagrams.py` | `figures/eda/methods_diagram.png` (Zoe's section roadmap), `figures/eda/roadmap.png` |
| `make_methods_intuition.py` | `figures/eda/methods_intuition.png` |
| `make_variance_ratio_v2.py` | `figures/varRatio/variance_ratio_panel.png` (slide 11 figure) |
| `make_data_overview.py`, `make_combined_data.py`, `make_kalshi_overview.py` | Data-overview figures for slides 4–5 |
| `make_methods_slides.py`, `insert_methods_slides.py` | Build / inject methods slides into the deck (legacy; current deck uses combined slides) |
| `build_presentation.py` | Full-deck regeneration helper |

---

## Methodological notes

- **Cluster bootstrap by contract** (Cameron, Gelbach & Miller 2008) is used for all within-contract Kalshi statistics so the CIs respect within-contract serial dependence.
- **Stationary block bootstrap** (Politis & Romano 1994) is used for spot return CIs in the variance ratio test, since spot is a continuous series with no natural cluster.
- **5-fold time-series cross-validation** (train on past, test on future) is used in the incremental information test; random shuffling is not permitted.
- **DeLong paired test** (DeLong, DeLong & Clarke-Pearson 1988) is the standard non-parametric test for AUC comparisons on shared evaluation data.
- **Each venue gets its natural feature set** in the incremental information test. The Kalshi feature set has more features (9 vs 8), so the asymmetry favors Kalshi — losing despite the asymmetry is a robust negative result.
