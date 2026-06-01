# figures/

Every figure in `Presentation-OQG-Banquet.pdf`, in the order it appears in the deck. Rendered in Times New Roman with text ≥ 32 pt (titles 36–38 pt) for 60-person-room legibility.

## Data overview

| File | Slide | Shows |
|---|---|---|
| `eda/data_joint_panel.png` | 4 | Joint panel: Coinbase BTC-USD close + Kalshi opening P(UP) + contracts/day across the 15 Feb – 28 Mar 2026 window (with the 13-day scraper outage marked) |

## Characterizing the raw P(UP) signal

| File | Slide | Shows |
|---|---|---|
| `eda/probability_convergence.png` | 5 | Mean P(UP) by minute-within-contract, split by realized outcome. Market resolves uncertainty monotonically toward 0/1 by minute 14 |
| `eda/hourly_accuracy.png` | 6 | Implied vs realized accuracy by UTC hour of contract open; eight hours show an overconfidence gap > 5 pp |
| `eda/accuracy_heatmap.png` | 7 | Accuracy by (minute, opening P(UP) bin); high opening conviction predicts high realized accuracy |

## Testing the signal against the underlying

| File | Slide | Shows |
|---|---|---|
| `eda/methods_diagram.png` | 8 | Three-diagnostic framework (lead-lag, variance ratio, incremental information) |
| `lead_lag.png` | 10 | Pearson cross-correlogram. Peak r = +0.51 at k = −2 — Kalshi lags BTC spot by ≈ 2 min; no positive lag is significant |
| `variance_ratio.png` | 12 | VR(q) for q ∈ {2, 3, 5, 7, 10}. Spot VR within 1% of 1.00; Kalshi P(UP) VR(2) = 1.24 rising to VR(10) = 1.41 — fails the random-walk null |
| `logReg/incremental_information.png` | 14 | ROC + accuracy table. Spot-only AUC = 0.796; Combined = 0.758 (DeLong p < 0.001 — adding Kalshi makes it *worse*) |

## Appendix

| File | Shows |
|---|---|
| `eda/calibration_panel.png` | Opening P(UP) distribution by outcome + calibration curve with 95% CIs |
| `eda/conviction_spread_accuracy.png` | Realized accuracy by opening conviction \|P(UP) − 0.5\| × 2 — high-conviction contracts (n ≪ low) trend higher |
| `logReg/feature_importance.png` | Standardized \|coefficient\| per feature across the three LRs — spot features dominate the Combined model |

## Supplementary

Not in the spoken deck. Extends the limitation and next-step from slide 15.

| File | Extends | Shows |
|---|---|---|
| `stress_events_lead_lag.png` | Slide-15 limitation (one volatility regime) | Lead-lag correlogram on top vs bottom RV decile of contracts. Peak at k = −2 holds across regimes — the 2-min lag is structural |
| `logReg/volatility_forecast.png` | Slide-15 next-step probe | HAR-RV vs HAR + GARCH(1,1) vs HAR + GARCH + Kalshi conviction spread for next-15-min realized vol. Conviction spread does not improve HAR-RV |

To regenerate any figure, see the [quickstart](../README.md#quickstart). Each script writes to the path above.
