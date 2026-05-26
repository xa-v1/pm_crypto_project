# Kalshi as an Alternative Data Source for BTC

**OQG Group Research Project — Spring 2026**

This repository characterizes Kalshi 15-minute BTC prediction-market contracts as an alternative data source for sentiment, and tests whether they carry directional information about the underlying BTC spot price beyond what is already reflected in the spot tape.

> **Headline finding:** Across three independent diagnostics, Kalshi P(UP) for BTC is a *calibrated, real-money, minute-frequency sentiment measurement that runs approximately 2 minutes behind the spot tape and exhibits positive serial correlation in its own changes.* It does **not** add incremental directional information above a same-window BTC spot baseline.

The presentation deck (`docs/presentation/`) and speaker notes (`docs/presentation/speaker_notes/`) are the primary deliverable; this README documents the analysis pipeline that produces them.

---

## Research question and motivation

Every Kalshi BTC contract resolves on Coinbase spot. For Kalshi P(UP) to be useful as a feature for trading the underlying asset, it must carry information *incremental* to the spot price process. We test this directly against three null hypotheses and report the result.

Xavi's section (slides 4–8) characterizes the within-Kalshi signal: calibration, intraday seasonality, accuracy convergence. Zoe's section (slides 9–13) tests Kalshi against the BTC spot benchmark using three diagnostics.

---

## The three diagnostics

| Slide | Diagnostic | Question | Result |
|---|---|---|---|
| 10 | **Lead-lag cross-correlogram** | At minute frequency, does P(UP) lead or lag the spot return? | Peak at lag k = −2, r = +0.51. Spot leads Kalshi by ~2 minutes; substantive reaction component bounded at ≥ 1 minute. |
| 11 | **Variance ratio test** *(Lo & MacKinlay 1988)* | Does P(UP) behave like a random walk on its own? | Spot does not reject the null (H = 0.498). Kalshi P(UP) rejects (H = 0.537; VR(2) = 1.24, VR(10) = 1.41). P(UP) under-reacts to its own information. |
| 12 | **Incremental information test** *(DeLong et al. 1988)* | Does augmenting a spot-only logistic regression with Kalshi features improve out-of-sample AUC? | Spot-only AUC = 0.796. Kalshi-only AUC = 0.690. Combined AUC = 0.758 — strictly worse than spot-only (DeLong p < 0.001). Kalshi features add no incremental signal. |

The three findings are consistent: Kalshi is downstream of spot and adds noise rather than signal when combined with spot features.

---

## Data

| File | Source | Window | Rows |
|---|---|---|---|
| `data/btc/kalshi_btc_prices.csv` | Kalshi public API | 15 Feb – 28 Mar 2026 | 33,951 minute-level rows over 2,264 BTC 15-min contracts |
| `data/btc/spot_btc_1m.csv` | Coinbase Exchange API | 15 Feb – 28 Mar 2026 | 58,948 BTC 1-min OHLCV bars |
| `data/btc/btc_contracts.csv` | Derived from raw Kalshi | — | One row per contract, with engineered features |
| `data/cleaned/merged_contracts.csv` | Kalshi × Coinbase | — | 1,916 matched contracts joined by exact UTC minute |
| `data/cleaned/*.json` | Diagnostic outputs | — | Result summaries for each test (lead-lag, VR, incremental info) |

ETH data is retained under `data/eth/` for reproducibility of an earlier cross-asset exploration; it is no longer used in the headline analysis.

---

## Repository layout

```
pm_crypto_project/
├── README.md                                        # this file
├── archive/                                         # out-of-scope code kept for reference
│   ├── README.md
│   └── polymarket/                                  # earlier Polymarket exploration
├── data/
│   ├── btc/                                         # Kalshi BTC, Coinbase BTC spot, contract features
│   ├── eth/                                         # ETH data (retained, not in headline analysis)
│   ├── cleaned/                                     # merged + diagnostic outputs (JSON, CSV)
│   └── README.md
├── docs/
│   ├── paper/                                       # whitepaper drafts (Word, PDF, Markdown)
│   ├── presentation/                                # decks + speaker notes
│   │   ├── presentation_d5.pptx                     # current version
│   │   └── speaker_notes/                           # speaker_notes.md is the source of truth
│   ├── sources/                                     # cited papers (Lo & MacKinlay, Wolfers & Zitzewitz, …)
│   └── README.md
├── figures/
│   ├── eda/                                         # Xavi's calibration, hourly, brier-score figures
│   ├── lagCorr/                                     # lead-lag cross-correlograms
│   ├── varRatio/                                    # variance ratio diagnostic
│   ├── logReg/                                      # incremental information test + appendix LR
│   └── README.md
├── scripts/
│   └── kalshi/
│       ├── btc/                                     # Kalshi BTC collection + cleaning
│       ├── eth/                                     # Kalshi ETH collection + cleaning
│       └── analysis/
│           ├── eda/                                 # data_exp.ipynb, preprocessing, Coinbase pull
│           ├── calibration/                         # Xavi's slides 6–8 analyses (accuracy, hourly, brier)
│           ├── lagCorr/                             # pm_spot_lead_lag.py and friends
│           ├── varRatio/                            # variance_ratio.py
│           ├── logReg/                              # incremental_information.py, logistic regression
│           ├── make_figures/                        # presentation-figure generators
│           └── README.md
└── .gitignore
```

---

## How to reproduce the headline diagnostics

From the repo root, after `pip install -r requirements.txt` (or your equivalent):

```bash
# 1. Pull/refresh spot data (idempotent; reads existing CSV if present)
python scripts/kalshi/analysis/eda/pull_coinbase_spot.py

# 2. Lead-lag cross-correlogram  →  data/cleaned/pm_spot_lead_lag.json, figures/lagCorr/lead_lag.png
python -m scripts.kalshi.analysis.lagCorr.pm_spot_lead_lag

# 3. Variance ratio test          →  data/cleaned/variance_ratio.json
python -m scripts.kalshi.analysis.varRatio.variance_ratio
python -m scripts.kalshi.analysis.make_figures.make_variance_ratio_panel    # figures/varRatio/variance_ratio_panel.png

# 4. Incremental information test →  data/cleaned/incremental_summary.json, figures/logReg/incremental_information.png
python -m scripts.kalshi.analysis.logReg.incremental_information
```

The presentation figures (with Times New Roman fonts sized for a 60-person room) are generated by scripts under `scripts/kalshi/analysis/make_figures/`.

---

## Appendix analyses (slides 19–22)

- **`scripts/kalshi/analysis/logReg/make_conviction_spread_accuracy.py`** — realized accuracy by Kalshi opening conviction bin (slide 20, Market Disagreement Signal).
- **`scripts/kalshi/analysis/logReg/make_lr_appendix_figures.py`** — feature-importance + 2-D decision-boundary diagnostics for the incremental-information LR (slides 21–22).
- **`scripts/kalshi/analysis/calibration/`** — Xavi's slides 8–9 work: hourly accuracy seasonality and (minute × conviction) accuracy heatmap.

---

## Limitations and next steps

See `docs/presentation/speaker_notes/speaker_notes.md` for the full discussion. Briefly:

- **Identification.** The 2-minute lag confounds Coinbase candle aggregation, Kalshi scraper averaging, and substantive reaction at Kalshi. Tick-level spot plus Kalshi order book data would identify each component.
- **Inference.** Cluster bootstrap by contract preserves within-contract dependence but not across-calendar-time dependence. A calendar-time block bootstrap would be more conservative.
- **Generalizability.** Six weeks of moderate-volatility data is one regime; replication during stress events (CPI releases, BTC-specific news shocks) is essential before treating the 2-minute lag and the 0.10 AUC gap as structural.
- **Most promising next step:** does the conviction spread `|P(UP) − 0.5|` forecast realized volatility above HAR-RV and GARCH benchmarks? The directional channel is dead, but the second-moment channel is open.

---

## References

Selected citations (full bibliography in `docs/sources/references.md`):

- Cameron, A. C., Gelbach, J. B., and Miller, D. L. (2008). Bootstrap-based improvements for inference with clustered errors. *Review of Economics and Statistics*, 90(3), 414–427. [DOI: 10.1162/rest.90.3.414](https://doi.org/10.1162/rest.90.3.414)
- Corsi, F. (2009). A simple approximate long-memory model of realized volatility. *Journal of Financial Econometrics*, 7(2), 174–196.
- DeLong, E. R., DeLong, D. M., and Clarke-Pearson, D. L. (1988). Comparing the areas under two or more correlated ROC curves. *Biometrics*, 44(3), 837–845. [DOI: 10.2307/2531595](https://doi.org/10.2307/2531595)
- Fama, E. F. (1970). Efficient capital markets: A review of theory and empirical work. *Journal of Finance*, 25(2), 383–417.
- Lo, A. W., and MacKinlay, A. C. (1988). Stock market prices do not follow random walks. *Review of Financial Studies*, 1(1), 41–66. [DOI: 10.1093/rfs/1.1.41](https://doi.org/10.1093/rfs/1.1.41)
- Wolfers, J., and Zitzewitz, E. (2004). Prediction markets. *Journal of Economic Perspectives*, 18(2), 107–126. [DOI: 10.1257/0895330041371321](https://doi.org/10.1257/0895330041371321)
