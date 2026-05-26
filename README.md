# Kalshi BTC Prediction Markets — OQG Research

Empirical characterization of Kalshi's 15-minute BTC prediction-market contracts. Three diagnostics test whether Kalshi P(UP) carries directional information *beyond* the Coinbase BTC spot tape:

| Diagnostic | Question | Finding |
|---|---|---|
| Lead-lag cross-correlogram | At minute frequency, does Kalshi P(UP) lead or lag spot? | Spot leads Kalshi by ≈ 2 min (peak r = +0.51 at k = −2) |
| Variance ratio (Lo & MacKinlay 1988) | Is Kalshi P(UP) a random walk? | Rejects null: VR(2) = 1.24, H = 0.537 |
| Incremental information (DeLong et al. 1988) | Does Kalshi add information above spot? | No: spot-only AUC = 0.796, combined = 0.758 |

Kalshi P(UP) is a calibrated, real-money sentiment measurement that runs ≈ 2 min behind the spot tape and adds no directional value above spot for short-horizon prediction.

---

## Layout

```
.
├── data/
│   ├── btc/        Kalshi BTC panel, Coinbase BTC spot, contract features
│   ├── eth/        Retained for reproducibility of earlier cross-asset work
│   └── cleaned/    Merged contracts and diagnostic outputs (JSON, CSV)
├── figures/        Outputs grouped by topic (eda/, logReg/) or flat for one-offs
├── scripts/
│   ├── analysis/   Source of every figure in the deck (run these)
│   ├── btc/        Xavi's original Kalshi BTC collection + exploratory scripts
│   └── eth/        Same for ETH
└── docs/           (gitignored)  Decks, speaker notes, paper drafts, cited PDFs
```

`scripts/analysis/` is the entrypoint. Xavi's `scripts/btc/` is preserved untouched for reference; the refactored, repo-root-relative versions of his accuracy/heatmap analyses live under `scripts/analysis/calibration/`.

---

## Quickstart

```bash
pip install -r requirements.txt
bash scripts/reproduce.sh                # all stages: data → diagnostics → appendix
# or pick a stage
bash scripts/reproduce.sh data
bash scripts/reproduce.sh diagnostics
bash scripts/reproduce.sh appendix
```

The appendix stage now also produces:
- `figures/logReg/volatility_forecast.png` — HAR-RV vs GARCH(1,1) vs Kalshi conviction spread for next-15-min realized vol
- `figures/stress_events_lead_lag.png` — lead-lag correlogram by realized-vol regime (top vs bottom decile)

Python 3.9+.

---

## Data

| File | Source | Window | Rows |
|---|---|---|---|
| `data/btc/kalshi_btc_prices.csv` | Kalshi public API | 15 Feb – 28 Mar 2026 | 33,951 minute-level rows over 2,264 contracts |
| `data/btc/spot_btc_1m.csv` | Coinbase Exchange API | same window | 58,948 BTC 1-min OHLCV bars |
| `data/cleaned/merged_contracts.csv` | Kalshi × Coinbase joined by exact UTC minute | — | 1,916 matched contracts |

---

## Methodological notes

- Cluster bootstrap by contract (Cameron, Gelbach & Miller 2008) for within-contract Kalshi CIs.
- Stationary block bootstrap (Politis & Romano 1994) for continuous spot CIs in the variance ratio test.
- 5-fold time-series cross-validation in the incremental information test — no random shuffling.
- DeLong paired test for AUCs on shared OOS data.
- Asymmetric feature design in the incremental info test (9 Kalshi vs 8 spot features); the asymmetry favors Kalshi.

---

## References

- Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2008). Bootstrap-based improvements for inference with clustered errors. *Review of Economics and Statistics*. [DOI](https://doi.org/10.1162/rest.90.3.414)
- DeLong, E. R., DeLong, D. M., & Clarke-Pearson, D. L. (1988). Comparing the areas under two or more correlated ROC curves. *Biometrics*. [DOI](https://doi.org/10.2307/2531595)
- Lo, A. W., & MacKinlay, A. C. (1988). Stock market prices do not follow random walks. *Review of Financial Studies*. [DOI](https://doi.org/10.1093/rfs/1.1.41)
- Wolfers, J., & Zitzewitz, E. (2004). Prediction markets. *Journal of Economic Perspectives*. [DOI](https://doi.org/10.1257/0895330041371321)
