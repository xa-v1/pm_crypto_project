# Kalshi Crypto Prediction Market Research

Predicting outcomes of 15-minute BTC and ETH binary contracts on Kalshi using machine learning and cross-asset market signals.

---

## Background

[Kalshi](https://kalshi.com) is a regulated prediction market exchange where participants trade yes/no contracts on real-world events. The contracts analyzed here resolve based on whether BTC or ETH price is **up or down** at the end of a 15-minute window. Each contract is priced as a probability: a contract trading at $0.62 implies the market estimates a 62% chance the outcome is YES (UP).

The central research question is: **can we build a model that extracts a systematic edge over the market's consensus probability?** If the market's opening probability were perfectly calibrated and instantly incorporated all public information, no edge would exist. We hypothesize that early intra-contract momentum and cross-asset signals carry information not yet fully priced in.

---

## Data

**Source:** Kalshi minute-level price feeds for BTC and ETH 15-minute binary contracts  
**Period:** February 16 – March 26, 2026  
**Raw volume:** ~34K minute-level BTC rows (2,264 contracts), ~29K ETH rows (1,943 contracts)

Each contract row records the asset, date, contract open time, strike threshold, P(UP), P(DOWN), and volume at that minute. The ticker encodes metadata — for example, `KXBTC15M-26FEB161830-30` means BTC, February 16 2026, opening at 18:30 UTC, $30 threshold.

**Data quality audits before any analysis:**
- Probability sums: |P(UP) + P(DOWN) − 1| ≤ 0.01 → 100% pass rate for both assets
- Duplicate (timestamp, ticker) pairs → none found
- Minimum contract length of 15 rows → one anomalous contract dropped per asset
- Final usable set: 2,263 BTC contracts (99.96%) and 1,942 ETH contracts (99.95%)

For cross-asset modeling, contracts were matched on `open_key` (timestamp floored to the minute), yielding **1,920 matched pairs** with 1,919 having both outcomes resolved.

---

## Analysis Pipeline

### 1. Feature Engineering

Rather than feeding raw minute-level data into models, we aggregate each 15-minute contract into a single row of summary statistics. This is the right choice here because: (a) the outcome is contract-level, not tick-level; (b) it avoids treating time-series within a contract as independent observations; and (c) it produces interpretable features.

**Features constructed per contract:**

| Feature | Description | Rationale |
|---|---|---|
| `opening_pup` | P(UP) at minute 0 | Market consensus at open; primary predictor baseline |
| `momentum_1min` | P(UP)[t=1] − P(UP)[t=0] | Earliest directional drift signal |
| `momentum_3min` | P(UP)[t=3] − P(UP)[t=0] | Sustained drift over first 3 minutes |
| `mean_pup_5min` | Mean P(UP) over first 5 minutes | Smoothed early conviction; reduces opening noise |
| `pup_vol_5min` | Std of P(UP) over first 5 minutes | Market uncertainty / disagreement proxy |
| `conviction_spread` | \|P(UP) − 0.5\| × 2 | How one-sided the opening probability is (range [0,1]) |
| `volume_log` | log(1 + opening volume) | Market activity; higher volume may signal stronger conviction |
| `hour_of_open` | UTC hour of contract open | Tests intraday seasonality in predictability |
| `prev_open_pup` | Previous contract's opening P(UP) | Tests autocorrelation / market memory |
| `opening_pup_eth` | ETH contract P(UP) at same time | Cross-asset signal (ETH as BTC predictor) |
| `momentum_3min_eth` | ETH 3-minute momentum | Cross-asset momentum signal |
| `btc_eth_divergence` | BTC opening P(UP) − ETH opening P(UP) | Divergence between assets (used in XGBoost only) |

The cross-asset features were motivated by the strong concurrent correlation between BTC and ETH markets (r = 0.78). If the two assets move together, early ETH price action may predict the BTC contract outcome and vice versa.

---

### 2. Exploratory Data Analysis

EDA was conducted before any modeling to build intuition and validate assumptions. Key findings:

**P(UP) distribution:** Approximately symmetric around 0.5 for both assets — the market does not systematically favor UP or DOWN, consistent with an efficient prediction market with thin directional biases.

**Market calibration:** The opening P(UP) is well-calibrated. In bins of [0.40–0.50], [0.50–0.60], [0.60–0.70], etc., the realized fraction of UP outcomes closely matches the predicted probability. This confirms the market is doing its job but does not rule out exploitable patterns.

**Signal evolution:** P(UP) drifts monotonically toward outcomes during the 15-minute window. Contracts that end UP show a steady increase from minute 0 to minute 14; contracts that end DOWN show a steady decrease. This confirms that early momentum is genuinely informative — the market is incorporating new information throughout the contract, not just at open.

**Market regime stability:** Rolling 50-contract accuracy hovers between 52–55% throughout the data period with no sharp breakpoints. The dataset is not contaminated by a structural regime shift.

**Disagreement signal:** Contracts where opening P(UP) is far from 0.5 (high conviction spread) are more accurately predicted by a simple threshold model. When the market is confident, it tends to be right. This supports using `conviction_spread` as a feature.

---

### 3. Lead-Lag Analysis

Before building cross-asset models, we tested whether BTC or ETH systematically *leads* the other. Using a cross-correlogram of P(UP) series at lags −3 to +3 minutes within matched contract windows:

- **Peak correlation at lag 0:** r = 0.781, p < 0.001
- **No significant lead-lag relationship** — the assets are driven by the same contemporaneous information rather than one reacting to the other with a delay

**Why this matters for modeling:** Lagged ETH features (e.g., ETH momentum predicting BTC 2 minutes later) would not add value. We therefore include only contemporaneous ETH signals.

---

### 4. Model Selection

Three models were evaluated, each building on the previous:

**Baseline Logistic Regression** uses only 3 same-asset BTC features: `opening_pup_btc`, `momentum_3min_btc`, `hour_of_open_btc`. This represents a minimal viable model using only the most direct market signals, and serves as the performance floor.

**Extended Logistic Regression** adds 8 engineered features plus 2 ETH cross-asset features (11 total). Logistic regression was chosen as the primary model because: (a) the dataset is not large enough to reliably support complex nonlinear models without overfitting; (b) coefficients are directly interpretable as log-odds; (c) it is well-understood and auditable for a research context. The `btc_eth_divergence` feature is excluded from LR because it is a linear combination of two included features, causing a singular design matrix.

**XGBoost** includes all 12 features including `btc_eth_divergence`, which gradient-boosted trees can handle without collinearity issues. XGBoost was chosen as the nonlinear benchmark because it handles feature interactions, requires minimal preprocessing, and provides SHAP interpretability. Conservative hyperparameters (shallow trees, low learning rate, subsampling) reduce overfitting on a ~1,600 sample OOS set.

Logistic regression and XGBoost were compared with a McNemar test (not just accuracy) to determine whether the prediction disagreements were statistically significant. They were not (p = 0.171), confirming that the datasets' modest size and feature set — not model complexity — is the binding constraint on performance.

---

### 5. Key Results

All models are evaluated out-of-sample on the same 1,596 contracts from the 5-fold time-series cross-validation. The majority-class baseline (always predict UP) achieves 52.5% accuracy.

| Model | Accuracy | AUC-ROC | Sharpe (TC=0) |
|---|---|---|---|
| Majority baseline | 52.5% | 0.500 | — |
| Baseline LR | 58.2% | 0.626 | +0.223 |
| Extended LR | **62.3%** | 0.684 | +0.578 |
| XGBoost | **63.5%** | 0.698 | +0.693 |

**Strongest predictors** (from LR coefficients and SHAP values):
1. `opening_pup_eth` — strongest single predictor of BTC outcome (β = +3.55, p < 0.001)
2. `momentum_3min_btc` — very strong same-asset signal (β = +6.34, p < 0.001)
3. `momentum_3min_eth` — cross-asset momentum (β = +4.69, p < 0.001)
4. `opening_pup_btc` — market consensus at open (β = +1.65, p = 0.033)

**The cross-asset signals are the main finding.** ETH opening P(UP) is a *stronger* predictor of BTC outcomes than BTC's own opening P(UP). This is consistent with the two assets sharing a common latent factor (crypto sentiment) that is sometimes priced faster or more strongly into ETH than BTC.

**Backtesting results** (threshold = 0.55, no transaction costs, $1 per trade):
- Extended LR: +$77.25 total PnL, 62.3% hit rate, Sharpe = +0.578
- Extended LR remains profitable up to ~0.2% per-trade transaction cost
- Sharpe peaks at thresholds 0.55–0.60; higher thresholds reduce trade count but improve hit rate

---

### 6. Sentiment Analysis (Exploratory)

`sentiment/btc_sentiment.py` collects Reddit posts from Bitcoin, CryptoCurrency, CryptoMarkets, and BitcoinMarkets every 15 minutes and scores them with **FinBERT**, a BERT variant fine-tuned on financial text. FinBERT was chosen over general-purpose sentiment models because it handles financial vocabulary (short, bearish, pullback, support level) far better than models trained on product reviews or news articles.

This component runs independently and outputs a timestamped CSV of sentiment scores. It is **not yet integrated** into the modeling pipeline and represents a potential future feature.

---

## Project Structure

```
pm_crypto_project/
├── data/                        # Raw and processed datasets, OOS prediction files
├── models/                      # Core analysis pipeline (run in order)
│   ├── preprocessing.py         # Data loading, validation, feature engineering, EDA
│   ├── lead_lag.py              # Cross-asset lead-lag correlation analysis
│   ├── logistic_regression.py   # Baseline + Extended LR with 5-fold time-series CV
│   ├── xgboost_model.py         # XGBoost classifier with SHAP interpretability
│   └── backtest.py              # Backtesting framework, benchmarks, TC sensitivity
├── figures/                     # All generated plots (EDA, model evaluation, backtest)
├── sentiment/                   # Optional real-time Reddit sentiment pipeline
│   └── btc_sentiment.py
├── scripts/                     # Historical data collection scripts (Kalshi, Polymarket)
└── docs/                        # Project background document
```

See [`models/README.md`](models/README.md) for detailed documentation of the modeling pipeline, preprocessing decisions, cross-validation methodology, evaluation statistics, and backtesting framework.

---

## How to Run

Run scripts in this order (each depends on outputs of the previous):

```bash
cd models/
python preprocessing.py       # generates data/btc_contracts.csv, eth_contracts.csv, merged_contracts.csv + EDA figures
python lead_lag.py            # generates data/lead_lag_result.json + cross_correlogram.png
python logistic_regression.py # generates data/lr_baseline_oos.csv, lr_extended_oos.csv + LR figures
python xgboost_model.py       # generates data/xgb_oos_predictions.csv + SHAP figures
python backtest.py            # generates all backtest figures; reads all OOS CSVs
```

Sentiment collection (optional, runs continuously):
```bash
cd sentiment/
pip install -r requirements.txt
python btc_sentiment.py
```

**Dependencies:** `pandas`, `numpy`, `scikit-learn`, `xgboost`, `shap`, `statsmodels`, `matplotlib`, `seaborn`  
**Python:** 3.9+

---

## Limitations

- **Short data window:** Two months (Feb–Mar 2026) is a thin training set. Generalization to different market regimes is untested.
- **Modest effect size:** 62–63% accuracy and Sharpe ~0.6–0.7 represent a real but not large edge. Transaction costs, slippage, and market impact can easily erode it.
- **No live deployment:** The pipeline is research-grade; it reads historical CSVs, not a live Kalshi API feed.
- **Sentiment not integrated:** FinBERT sentiment is collected but untested as a model feature.
- **Static model:** No rolling retraining. If market dynamics shift, model performance may degrade without periodic refitting.