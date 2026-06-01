# Kalshi BTC Prediction Markets as a Quant Data Source

**OQG research project — Spring 2026.** Empirical characterization of Kalshi's 15-minute Bitcoin prediction-market contracts. Companion code to `Presentation-OQG-Banquet.pdf` (repo root).

> **Research question.** Prediction markets aggregate dispersed private information into a price (Wolfers & Zitzewitz, 2004). If that holds, do short-horizon crypto contracts on platforms like Kalshi generate a tradeable probability signal, or do they simply reflect what BTC spot already knows? No prior work characterizes intraday crypto event contracts as a standalone data source.

---

## Story of the analysis

### 1. Data

Six weeks (15 Feb – 28 Mar 2026) of:

| Source | What | Window | Volume |
|---|---|---|---|
| Kalshi public API | 15-min BTC P(UP) contracts at minute resolution | 15 Feb – 28 Mar 2026 | 33,951 minute-level rows over 2,264 contracts |
| Coinbase Exchange API | BTC-USD 1-min OHLCV bars | same | 58,948 bars |
| Joined | Inner join on exact UTC minute | — | 1,916 matched contracts |

### 2. Characterizing the raw signal

Before asking what Kalshi P(UP) carries, we asked how it behaves:

- **Probability convergence within contracts** — P(UP) starts near 50/50 and resolves monotonically toward 0/1 by minute 14. Implication: any predictive signal has to come from the early window or we conflate convergence with information.
- **Intraday variability in realized accuracy** — Implied accuracy hovers around 55% all day. Realized accuracy varies sharply; eight hours show an overconfidence gap exceeding 5 percentage points.
- **Accuracy by opening conviction** — Within any contract minute, accuracy rises with opening conviction. Contracts opening at P(UP) ≥ 0.75 are right 100% of the time from minute 0; near-50/50 contracts only cross 60% accuracy by minute 2.

### 3. Three diagnostics: does Kalshi P(UP) carry information beyond BTC spot?

| Test | Question | Result |
|---|---|---|
| **Lead-lag cross-correlogram** (Cameron, Gelbach & Miller 2008) | At minute frequency, does Kalshi lead or lag spot? | Spot leads Kalshi by ≈ 2 min. Peak r = +0.51 at k = −2; no positive lag is significant. |
| **Variance ratio test** (Lo & MacKinlay 1988) | Is P(UP) a random walk? | Rejects the null. P(UP) VR(2) = 1.24, VR(10) = 1.41 — under-reaction. Spot VR(q) stays within 1% of 1.00. |
| **Incremental information** (DeLong et al. 1988) | Does Kalshi add directional info above spot? | No. Spot-only AUC = 0.796 (73.1% accuracy). Combined AUC = 0.758 — *worse*. DeLong p < 0.001. |

### 4. Verdict

Kalshi P(UP) is a calibrated, real-money sentiment measurement that runs ≈ 2 min behind the spot tape and adds no directional value above spot for short-horizon prediction.

### 5. Limitations & next steps

The presented analysis covers one volatility regime (six weeks of moderate vol). Two follow-up probes live in this repo as supplementary research (not in the spoken deck):

- **Stress-event replication of the lead-lag diagnostic** — rerun on top vs bottom RV decile of contracts. The 2-min lag holds across regimes (full r = 0.51, stress r = 0.53, quiet r = 0.51), suggesting the lag is structural.
- **Second-moment channel** — does the conviction spread |P(UP) − 0.5| forecast next-15-min realized vol? Benchmarked against HAR-RV (Corsi 2009) and GARCH(1,1). Conviction spread does not improve HAR-RV out-of-sample.

---

## Layout

```
.
├── Presentation-OQG-Banquet.pdf       The deck
├── data/
│   ├── btc/        Kalshi BTC panel, Coinbase BTC spot, contract features
│   ├── eth/        Same files for ETH (Xavi's earlier cross-asset proof-of-concept)
│   └── cleaned/    Merged contracts + JSON outputs of each diagnostic
├── figures/        Every figure in the deck, grouped by deck section
├── scripts/
│   ├── analysis/   Source of every deck figure — the analytical pipeline
│   ├── btc/        Xavi's original Kalshi BTC collection + exploratory work
│   └── eth/        Same for ETH
└── docs/           (gitignored)  Decks, speaker notes, paper drafts, cited PDFs
```

Xavi's `scripts/btc/` and `scripts/eth/` are preserved as the original data-collection pipeline and cross-asset proof-of-concept. The refactored, repo-root-relative versions of his calibration analyses live under `scripts/analysis/calibration/`.

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

Python 3.9+.

---

## References

- Wolfers, J., & Zitzewitz, E. (2004). Prediction markets. *Journal of Economic Perspectives*. [DOI](https://doi.org/10.1257/0895330041371321)
- Le, N. A. (2026). Decomposing crowd wisdom: domain-specific calibration dynamics in prediction markets. *arXiv:2602.19520*.
- Mohanty, H., & Krishnamachari, B. (2026). Do prediction markets forecast cryptocurrency volatility? Evidence from Kalshi macro contracts. *arXiv:2604.01431*.
- Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2008). Bootstrap-based improvements for inference with clustered errors. *Review of Economics and Statistics*. [DOI](https://doi.org/10.1162/rest.90.3.414)
- Lo, A. W., & MacKinlay, A. C. (1988). Stock market prices do not follow random walks. *Review of Financial Studies*. [DOI](https://doi.org/10.1093/rfs/1.1.41)
- DeLong, E. R., DeLong, D. M., & Clarke-Pearson, D. L. (1988). Comparing the areas under two or more correlated ROC curves. *Biometrics*. [DOI](https://doi.org/10.2307/2531595)
