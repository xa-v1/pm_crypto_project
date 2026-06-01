# scripts/analysis/

Source of every figure in `Presentation-OQG-Banquet.pdf`. Ordered by where each script lands in the deck's narrative.

Each script reads from `data/btc/` and writes JSON or CSV outputs to `data/cleaned/` and figures to `figures/`.

## Stage 1 — Data assembly

The two raw sources are joined into a per-contract panel.

| Script | What it does | Output |
|---|---|---|
| `eda/pull_coinbase_spot.py` | Fetches Coinbase BTC-USD and ETH-USD 1-min OHLCV. | `data/btc/spot_btc_1m.csv`, `data/eth/spot_eth_1m.csv` |
| `eda/preprocessing.py` | Joins Kalshi minute panel × Coinbase spot on UTC minute; builds per-contract feature CSV. Also renders deck slide 5 (within-contract convergence) and the appendix Calibration panel. | `data/cleaned/merged_contracts.csv`, `figures/eda/probability_convergence.png`, `figures/eda/calibration_panel.png` |
| `make_figures/make_data_joint_panel.py` | Joint price-and-P(UP) panel showing the six-week window and the scraper outage. | `figures/eda/data_joint_panel.png` (slide 4) |

## Stage 2 — Characterizing the raw P(UP) signal

Before testing P(UP) against spot, the deck profiles the Kalshi signal on its own (slides 5–7).

| Script | What it does | Output |
|---|---|---|
| `eda/preprocessing.py` | Within-contract probability convergence: P(UP) starts ≈ 50/50 and resolves monotonically by minute 14. | `figures/eda/probability_convergence.png` (slide 5) |
| `calibration/make_hourly_accuracy.py` | Implied vs realized accuracy by UTC hour of contract open; flags overconfidence gaps > 5 pp. | `figures/eda/hourly_accuracy.png` (slide 6) |
| `calibration/make_accuracy_heatmap.py` | Accuracy by (minute-within-contract, opening P(UP) bin) — shows opening conviction predicts realized accuracy. | `figures/eda/accuracy_heatmap.png` (slide 7) |

## Stage 3 — Testing the signal against the underlying

Slide 8 frames three diagnostics; slides 9–14 deliver them.

| Script | What it does | Output | Slide |
|---|---|---|---|
| `make_figures/make_diagrams.py` | The three-test methods diagram. | `figures/eda/methods_diagram.png` | 8 |
| `lead_lag.py` | Pearson cross-correlogram of ΔP(UP) vs spot 1-min log-return at lag k ∈ [−10, +10], with cluster-bootstrap 95% CIs (Cameron, Gelbach & Miller 2008). | `data/cleaned/pm_spot_lead_lag.json`, `figures/lead_lag.png` | 10 |
| `variance_ratio.py` | Lo–MacKinlay (1988) VR(q) for P(UP) vs spot at q ∈ {2, 3, 5, 7, 10}. Cluster bootstrap for P(UP); stationary block bootstrap (Politis & Romano 1994) for spot. | `data/cleaned/variance_ratio.json` | 12 (data) |
| `make_figures/make_variance_ratio_panel.py` | Plots both VR curves with 95% CIs on a shared axis. | `figures/variance_ratio.png` | 12 |
| `logReg/incremental_information.py` | The headline incremental-information test. Three nested LRs (Spot-only / Kalshi-only / Combined) with 5-fold time-series CV; bootstrap AUC CIs; DeLong (1988) paired test on shared OOS contracts. | `data/cleaned/incremental_summary.json`, `data/cleaned/incremental_oos.csv`, `figures/logReg/incremental_information.png` | 14 |

## Stage 4 — Appendix figures

These show in the appendix when there is time.

| Script | What it does | Output |
|---|---|---|
| `logReg/make_conviction_spread_accuracy.py` | Realized accuracy as a function of opening conviction \|P(UP) − 0.5\| × 2. | `figures/eda/conviction_spread_accuracy.png` |
| `logReg/make_lr_appendix_figures.py` | Standardized \|coefficient\| per feature for each of the three LRs — shows spot features dominate the Combined model. | `figures/logReg/feature_importance.png` |

## Stage 5 — Supplementary research

Two follow-up probes that extend the deck's stated limitation (slide 15) and next-step idea. Not in the spoken deck; included here so the verdict and the proposed extensions are reproducible.

| Script | Extends | Output |
|---|---|---|
| `stress_events.py` | The slide-15 limitation that we only observed one volatility regime. Reruns the lead-lag correlogram on the top vs bottom RV decile of contracts. The 2-min lag holds across regimes. | `data/cleaned/stress_events.json`, `figures/stress_events_lead_lag.png` |
| `logReg/make_volatility_forecast.py` | The slide-15 next-step probe: can the conviction spread \|P(UP) − 0.5\| forecast next-15-min realized vol above HAR-RV (Corsi 2009) and GARCH(1,1)? | `data/cleaned/volatility_forecast.json`, `figures/logReg/volatility_forecast.png` |

## Methodological notes

- **Cluster bootstrap by contract** (Cameron, Gelbach & Miller 2008) for within-contract Kalshi statistics — accounts for the fact that minute-level observations within a contract are not independent.
- **Stationary block bootstrap** (Politis & Romano 1994) for continuous spot CIs in the VR test.
- **5-fold time-series CV** (train past, test future) in the incremental information test — no random shuffling.
- **DeLong paired test** (1988) compares AUCs on the shared OOS panel rather than across resampled folds.
- The two feature sets are each venue's natural set (8 spot vs 9 Kalshi); the asymmetry favors Kalshi.
