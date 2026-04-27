# Models: Methodology Reference

This document covers the modeling pipeline in detail — feature engineering, preprocessing decisions, cross-validation design, evaluation metrics, statistical tests, and the backtesting framework. It assumes familiarity with general data science and quantitative finance but explains every choice made here and why.

---

## Table of Contents

1. [Preprocessing & Feature Engineering](#1-preprocessing--feature-engineering)
2. [Model Architecture Choices](#2-model-architecture-choices)
3. [Cross-Validation Design](#3-cross-validation-design)
4. [Evaluation Metrics](#4-evaluation-metrics)
5. [Statistical Tests](#5-statistical-tests)
6. [Backtesting Framework](#6-backtesting-framework)

---

## 1. Preprocessing & Feature Engineering

### Raw Data Format

The raw input is minute-level Kalshi data: one row per minute per contract, recording P(UP), P(DOWN), volume, and a ticker that encodes all contract metadata.

**Contract identification:** The ticker `KXBTC15M-26FEB161830-30` is parsed as:
- Asset: BTC
- Date: February 16, 2026
- Open time: 18:30 UTC
- Strike: $30 threshold

We parse the ticker to extract `open_key` (the contract open timestamp), which is used as the join key for cross-asset matching.

### Why Aggregate to Contract Level?

The natural unit of prediction is the contract, not the minute. A minute-level model would need to treat each row as independent, which is wrong — rows within a contract are highly autocorrelated. Aggregating to contract level gives us one row per outcome and forces features to summarize the contract's early trajectory, which is the actual information available to a trader who sees the contract open and must decide whether to trade.

### Data Quality Checks

Applied before any feature computation:

1. **Probability consistency:** Assert |P(UP) + P(DOWN) − 1| ≤ 0.01. Kalshi guarantees this for live data; checking here catches any scraping artifacts.
2. **Duplicate rows:** Assert no duplicate (timestamp, ticker) pairs.
3. **Minimum contract length:** Drop contracts with fewer than 15 rows (incomplete data). One contract per asset was dropped.

These checks run programmatically in `preprocessing.py` and print a summary. If any check fails, the script raises an assertion error rather than silently producing bad features.

### Feature Engineering Decisions

Every feature is computed from **only the first few minutes of the contract** (typically first 5 minutes out of 15). This is intentional and critical: a real trader sees only the opening dynamics before committing to a trade. Using minute 10 or 14 data to predict the minute 15 outcome would be data leakage.

**`opening_pup`** — P(UP) at minute 0. This is the market's opening consensus and the most direct signal. It is well-calibrated (see EDA), meaning a 0.65 opening probability really does correspond to ~65% UP outcomes. However, calibration alone does not create an edge; it just means the market is rational. We include this because even a well-calibrated probability is a strong predictor.

**`momentum_1min` and `momentum_3min`** — The change in P(UP) from minute 0 to minute 1 or 3. These capture early information arrival. If P(UP) jumps from 0.52 to 0.61 in the first 3 minutes, that directional drift has predictive value beyond the opening probability alone — it suggests the market is incorporating new information (a price move in spot BTC, a news event, etc.) in a consistent direction. This is the most statistically significant feature in the logistic regression.

**`mean_pup_5min`** — Mean P(UP) over minutes 0–4. Smooths out opening noise. Some contracts open with erratic first-minute probabilities due to thin early liquidity. A 5-minute average is a more stable representation of early market conviction than the single opening tick.

**`pup_vol_5min`** — Standard deviation of P(UP) over the first 5 minutes. High volatility in P(UP) during the opening minutes means the market is uncertain or actively repricing — a sign of lower predictability. Low volatility means the market settled quickly on a consensus.

**`conviction_spread`** — |P(UP) − 0.5| × 2, scaled to [0, 1]. A spread of 0 means the market is exactly 50/50; a spread of 1 means the market is 100% confident. This feature captures the one-sidedness of the market independent of direction, and is a strong predictor of whether simple threshold strategies will work on this contract.

**`volume_log`** — log(1 + opening volume). Log-transformed because volume distributions are right-skewed. Higher opening volume may indicate stronger conviction or more informed participation at that time.

**`hour_of_open`** — UTC hour. Kalshi markets are most active during US trading hours. Intraday patterns in predictability (e.g., markets may be more one-sided near major economic releases at 8:30 AM ET = 13:30 UTC) are captured here.

**`prev_open_pup`** — The previous contract's opening P(UP). Tests whether markets exhibit memory — whether a bullish opening in one 15-minute window predicts a bullish opening in the next. This is the autocorrelation signal. It turned out to be weak but was included to test the hypothesis.

**Cross-asset features (`opening_pup_eth`, `momentum_3min_eth`)** — ETH P(UP) and ETH momentum at the same time period. These are the key finding of the project: ETH signals predict BTC outcomes better than same-asset BTC signals alone. This makes sense because BTC and ETH share a common underlying crypto sentiment factor; sometimes that sentiment is priced first or more strongly in one asset.

**`btc_eth_divergence`** — BTC opening P(UP) minus ETH opening P(UP). Measures how much the two markets disagree at open. Excluded from logistic regression because it is a linear combination of `opening_pup_btc` and `opening_pup_eth`, causing perfect collinearity (singular design matrix). Included in XGBoost, where trees handle collinear features without numerical issues.

### Standardization

Features are standardized (zero mean, unit variance via `StandardScaler`) before fitting logistic regression. This is required for two reasons:

1. **Interpretability:** Standardized logistic regression coefficients are directly comparable in magnitude — a larger absolute coefficient means a stronger predictor on the same scale.
2. **Numerical stability:** `lbfgs` optimizer converges faster and more reliably on standardized inputs.

XGBoost does not require standardization (tree splits are scale-invariant), but the same StandardScaler from the training fold is applied to keep the pipeline consistent. The scaler is fit **only on the training fold** within each CV iteration to prevent leakage.

### Cross-Asset Merging

BTC and ETH contracts are merged on `open_key`. This inner join retains only contracts where both assets have data for the same 15-minute window, dropping 42 contracts (mostly at data collection boundaries). The resulting merged dataset of 1,919 contracts is used for all cross-asset model training and evaluation.

---

## 2. Model Architecture Choices

### Logistic Regression

Logistic regression was chosen as the primary model for this dataset for several reasons:

- **Sample size discipline:** ~1,600 OOS contracts is not a large dataset by ML standards. Complex models with many parameters risk overfitting, which time-series CV will detect but not fix.
- **Interpretability:** Coefficients directly quantify the log-odds contribution of each feature. This is a research project — understanding *why* the model works matters as much as *that* it works.
- **Calibrated probabilities:** Logistic regression produces well-calibrated probability outputs, making downstream threshold analysis and backtesting more meaningful.
- **Auditable:** The regularization (default L2 via sklearn's `C=1.0`) and solver (`lbfgs`, quasi-Newton) are standard and well-understood.

We used `statsmodels.Logit` on the full dataset after CV to obtain proper p-values and confidence intervals on coefficients. Sklearn's `LogisticRegression` does not provide p-values natively.

### XGBoost

XGBoost was included as the nonlinear benchmark:

- **Handles interactions and nonlinearity:** If the effect of `momentum_3min` depends on `opening_pup` in a nonlinear way, logistic regression cannot capture this without manual interaction terms. XGBoost does this automatically via tree splits.
- **Handles the collinear divergence feature:** Trees do not require orthogonal features.
- **SHAP interpretability:** XGBoost + SHAP gives a rigorous, model-agnostic decomposition of each prediction into feature contributions, allowing direct comparison with LR coefficients.
- **Conservative hyperparameters to reduce overfitting:**
  - `n_estimators=300, max_depth=4` — shallow trees, many iterations; avoids memorizing individual contracts
  - `learning_rate=0.05` — slower learning, more regularized
  - `subsample=0.8, colsample_bytree=0.8` — feature and row subsampling (stochastic gradient boosting)
  - `min_child_weight=5` — requires at least 5 samples per leaf; prevents splits on noise

The McNemar test result (p = 0.171) tells us XGBoost's marginal improvement over Extended LR is not statistically significant. The dataset, not the model, is the bottleneck.

---

## 3. Cross-Validation Design

### Why TimeSeriesSplit, Not Standard K-Fold

Standard k-fold cross-validation randomly shuffles samples into folds. This is **wrong for time-series data** because it allows the model to train on future data and test on past data. In a prediction market context this is catastrophic — the model could learn from a Monday contract to predict the Friday contract from the previous week. Any reported accuracy from shuffled CV would be inflated and non-deployable.

`TimeSeriesSplit` with `n_splits=5` creates 5 folds where each training set consists exclusively of contracts *before* the test set in time:

```
Fold 1: Train [t=1...320]   → Test [t=321...640]
Fold 2: Train [t=1...640]   → Test [t=641...960]
Fold 3: Train [t=1...960]   → Test [t=961...1280]
Fold 4: Train [t=1...1280]  → Test [t=1281...1600]
Fold 5: Train [t=1...1600]  → Test [t=1601...1919]
```

(Approximate — exact splits depend on dataset size.)

OOS predictions from the last fold (or averaged across folds) become the evaluation set. Using predictions only from folds where the test period strictly follows the training period guarantees no look-ahead bias.

### Scaler Fit Inside CV Loop

The `StandardScaler` is fit **inside each fold** on the training data only, then applied to both train and test. Fitting the scaler on the full dataset before CV is a common mistake that leaks information about future test distributions into the standardization. Our pipeline handles this correctly.

### Train-Test Split for OOS Reporting

The final 1,596-contract OOS set used in all reported metrics comes from concatenating test-fold predictions across the 5 time-series folds. This gives the largest possible OOS evaluation without any future leakage.

---

## 4. Evaluation Metrics

For each metric: what it measures, how it is computed, and why we care about it for this problem.

### Accuracy

**Formula:** (TP + TN) / (TP + TN + FP + FN)

The fraction of contracts predicted correctly. The baseline to beat is 52.5% (the fraction of contracts that end UP — always predicting UP would score 52.5%). A model must beat this to have any value.

**Limitation:** Accuracy ignores the asymmetry of prediction errors. In a trading context, a false positive (predict UP, outcome DOWN) and a false negative (predict UP, outcome DOWN when you held flat) have different economic consequences. Use alongside precision and recall.

### Precision

**Formula:** TP / (TP + FP) — of all contracts predicted UP, what fraction actually went UP?

In trading terms: hit rate on long bets. If precision is 0.60, 60% of your long trades win. This is the most economically meaningful single metric for a simple long-only strategy.

### Recall

**Formula:** TP / (TP + FN) — of all contracts that actually went UP, what fraction did we predict correctly?

In trading terms: coverage. High recall means you captured most of the winning UP trades; low recall means you were too conservative and left many profitable opportunities on the table. The baseline LR has recall 0.691 vs extended LR's 0.758 — the extended model catches more winning trades.

### F1 Score

**Formula:** 2 × (Precision × Recall) / (Precision + Recall)

The harmonic mean of precision and recall, useful when you care about both simultaneously. The harmonic mean penalizes extreme imbalances — a model with 1.0 precision and 0.01 recall would have F1 = 0.02, not 0.505. We report this alongside individual precision and recall.

### AUC-ROC

**What it is:** The area under the Receiver Operating Characteristic curve. The ROC curve plots True Positive Rate (recall) vs. False Positive Rate at every possible classification threshold.

**Interpretation:** AUC is the probability that the model ranks a randomly chosen UP contract higher than a randomly chosen DOWN contract. AUC = 0.5 is random; AUC = 1.0 is perfect. Our extended LR achieves AUC = 0.684, meaning 68.4% of the time it correctly ranks a UP contract above a DOWN contract.

**Why it matters here:** AUC is threshold-independent — it evaluates the quality of the probability ordering across all possible trading thresholds, not just at 0.5. Since backtesting involves scanning thresholds, AUC is a better summary of model quality than accuracy at a single threshold.

### PR-AUC (Precision-Recall AUC)

The area under the Precision-Recall curve. While less commonly reported, PR-AUC is more informative than ROC-AUC when classes are imbalanced, because it focuses on the positive class (UP) and does not reward true negatives. With a 52.5% base rate our data is close to balanced, so AUC-ROC and PR-AUC tell similar stories, but we report both.

### Calibration

Beyond classification metrics, we examine whether predicted probabilities are meaningful. A model predicting 0.65 for UP should be right ~65% of the time. We visualize this with a reliability diagram (calibration plot). Our models inherit good calibration from the well-calibrated opening P(UP) inputs; logistic regression generally preserves this calibration; XGBoost can lose it and may benefit from Platt scaling in future work.

---

## 5. Statistical Tests

### McNemar Test

The McNemar test compares two binary classifiers on the **same test set** by focusing on disagreements. It is more appropriate than comparing raw accuracy numbers because it accounts for the correlation structure between classifiers evaluated on the same samples.

**Setup:**
- Build a 2×2 contingency table of prediction disagreements between Model A and Model B
- Cell (A correct, B wrong) = n₁₀; Cell (A wrong, B correct) = n₀₁
- Under H₀ (models are equivalent): n₁₀ and n₀₁ should be equal

**Statistic:** χ² = (|n₁₀ − n₀₁| − 1)² / (n₁₀ + n₀₁), with 1 degree of freedom

**Result:** XGBoost vs Extended LR: χ² = 1.872, p = 0.171 — we fail to reject H₀. The models disagree on ~some contracts, but neither systematically outperforms the other often enough to be confident the difference is real rather than noise. **Conclusion:** XGBoost's 1.2% accuracy advantage over Extended LR is not statistically significant on this dataset.

### Logistic Regression Coefficient Significance (statsmodels)

We re-fit the Extended LR on the full dataset using `statsmodels.Logit` to obtain proper coefficient standard errors and p-values via the z-test (large-sample approximation to the t-distribution).

**Key significant coefficients:**
- `momentum_3min_btc`: β = +6.34, z = 4.3, p < 0.001 — strongest same-asset signal
- `opening_pup_eth`: β = +3.55, z ≈ 3.7, p < 0.001 — strongest cross-asset signal
- `momentum_3min_eth`: β = +4.69, z ≈ 3.5, p < 0.001 — cross-asset momentum
- `opening_pup_btc`: β = +1.65, z ≈ 2.1, p = 0.033 — significant but weaker

Note that these p-values are computed on the full dataset (including training data), so they measure feature significance under the full data distribution — not OOS predictive power specifically. The CV accuracy is the correct measure of OOS performance.

### SHAP Values (XGBoost)

SHAP (SHapley Additive exPlanations) decomposes each prediction into additive contributions from each feature. Unlike feature importances from tree splits (which only show which features are used, not direction or magnitude), SHAP values show exactly how much each feature pushed a specific prediction above or below the baseline.

**SHAP bar chart** (`xgb_shap_bar.png`): Mean |SHAP value| across all predictions — global feature importance

**SHAP beeswarm** (`xgb_shap_beeswarm.png`): Each dot is one prediction; x-axis is SHAP value (positive = pushes toward UP), color is feature value (red = high, blue = low). This lets you see: "high `momentum_3min_btc` strongly pushes toward UP, low `momentum_3min_btc` strongly pushes toward DOWN."

SHAP results for this dataset align closely with LR coefficients, reinforcing that the relationships are approximately linear and the primary model (Extended LR) is capturing the key structure.

### Lead-Lag Significance

The cross-correlogram in `lead_lag.py` tests each lag's Pearson correlation with a t-test adjusted for the effective sample size (reduced by autocorrelation in the series). The peak at lag 0 (r = 0.781) is highly significant (p < 0.001). Lags ±1 through ±3 are smaller and the differences from lag 0 are not significant, confirming the concurrent relationship is the dominant one.

---

## 6. Backtesting Framework

This section explains backtesting from first principles, then details our specific implementation.

### What Backtesting Is and Why It Matters

A model can achieve 62% accuracy out-of-sample in cross-validation and still be useless for trading if that accuracy doesn't translate into consistent profits after accounting for transaction costs, realistic trade sizing, and the actual distribution of when trades are taken.

Backtesting simulates what would have happened if you had deployed the model historically. It answers: *given model predictions on the OOS data, what would a specific trading strategy have earned?*

The key discipline: **backtesting must use only OOS predictions** (from the CV pipeline), never predictions from a model trained on the same data being evaluated. Our backtest reads `lr_baseline_oos.csv`, `lr_extended_oos.csv`, and `xgb_oos_predictions.csv` — predictions the model had never seen during training.

### Strategy Logic

The strategy is a simple **binary long/short/flat** based on the model's predicted probability of UP:

```
if predicted_prob_UP > threshold:      → LONG (bet on UP)
if predicted_prob_UP < 1 - threshold:  → SHORT (bet on DOWN)
else:                                  → FLAT (no trade)
```

For example, with `threshold = 0.55`:
- Predicted P(UP) = 0.62 → LONG
- Predicted P(UP) = 0.41 → SHORT (because 0.41 < 0.45 = 1 − 0.55)
- Predicted P(UP) = 0.51 → FLAT

**Threshold parameter:** The threshold controls the aggressiveness of the strategy. A low threshold (0.50) trades every contract; a high threshold (0.70) only trades when the model is very confident. We scan thresholds from 0.50 to 0.70 in 0.05 steps to find the Sharpe-optimal threshold.

### PnL Calculation

Each Kalshi binary contract pays +$1 for correct prediction, -$1 for incorrect (before transaction costs). We model this as:

```
if signal == LONG and outcome == UP:    PnL = +1
if signal == LONG and outcome == DOWN:  PnL = -1
if signal == SHORT and outcome == DOWN: PnL = +1
if signal == SHORT and outcome == UP:   PnL = -1
if signal == FLAT:                      PnL = 0
```

This is the idealized binary payoff of a prediction market contract. In practice, actual Kalshi payoffs depend on the price at which you buy the contract — a contract bought at $0.55 pays $0.45 profit if correct and loses $0.55 if wrong. Our simplified +/-$1 model is equivalent to always buying contracts at $0.50, which is an approximation.

**Portfolio model:** $1,000 starting capital, $1 at risk per trade (0.1% of capital per trade). This sizing keeps losses small relative to capital. The portfolio value evolves as:

```
portfolio[t] = portfolio[t-1] + (PnL[t] − transaction_cost) × position_size
```

### Transaction Costs

Real trading involves three types of cost:
1. **Bid-ask spread:** The difference between the best bid and ask. On Kalshi, this is typically 1–4 cents on a $0.50 contract (2–8%).
2. **Exchange fees:** Kalshi charges a percentage of winnings.
3. **Market impact:** Large orders move the price against you. At our trade size this is negligible.

We model transaction costs as a flat percentage of the trade value deducted whether you win or lose. The backtester scans TC rates of 0%, 0.1%, 0.5%, and 1.0% to show breakeven cost levels.

**Result:** Extended LR remains profitable (Sharpe > 0) up to approximately 0.20% per-trade TC. At 0.5% TC, the strategy breaks even. This is tight — Kalshi's spreads may exceed 0.2% on less liquid contracts, which means real performance could be significantly worse than TC=0 results.

### Performance Metrics

**Hit Rate** — Fraction of trades (excluding FLAT) that were correct. This is precision averaged over both LONG and SHORT signals. Our extended LR achieves ~62% hit rate at threshold 0.55.

**Total PnL** — Sum of per-trade profits. Extended LR: +$77.25 over 1,596 trades ≈ $0.048 per trade on average.

**Sharpe Ratio** — Risk-adjusted return, the standard measure of strategy quality:

```
Sharpe = (Mean period return) / (Std of period returns) × √(periods per year)
```

For annualization: Kalshi contracts are 15-minute intervals. Assuming 24/7 operation:

```
periods per year = 24 hours × 4 contracts/hour × 365 days = 35,040
```

However, we observe the market is most active during US trading hours. Using 252 trading days × 26 periods/day = 6,552 as a conservative estimate of actively traded periods. Our Extended LR achieves annualized Sharpe ≈ +0.578.

**Interpretation of Sharpe:** A Sharpe > 1.0 is generally considered good for a systematic strategy. Our Sharpe of 0.58 is positive and meaningful, but below this bar — consistent with a real but modest edge. For reference, SPY (S&P 500) historically achieves Sharpe ~0.5–0.7, so our model is in that range despite operating on much shorter timescales.

**Maximum Drawdown (Max DD)** — The largest peak-to-trough decline in cumulative portfolio value during the backtest period. Extended LR max DD = −$31.10. This measures the worst losing streak the strategy experienced. A low max DD relative to total PnL (31/77 ≈ 40%) is acceptable for a 2-month backtest.

**Drawdown formula:**
```
running_max[t] = max(portfolio[0], ..., portfolio[t])
drawdown[t] = portfolio[t] - running_max[t]
max_drawdown = min(drawdown)
```

### Threshold Analysis

The `backtest_sharpe_grid.png` figure shows Sharpe ratio and trade count as a function of threshold. Key relationships:

- **Threshold 0.50:** All contracts traded; most trades, lowest per-trade accuracy, Sharpe may be negative if model is weak
- **Threshold 0.55–0.60:** Optimal range for Extended LR and XGBoost; filters out low-conviction contracts while maintaining sufficient trade count
- **Threshold 0.65–0.70:** Very few trades; high per-trade accuracy but high variance (few samples); Sharpe estimates unreliable

The optimal threshold is where the model's confidence is meaningfully predictive without reducing trade count so much that statistical estimates become noisy.

### Benchmark Comparisons

The backtest compares strategies against two passive benchmarks:

**Buy-and-hold SPY** — Total return of S&P 500 over the same Feb 16 – Mar 26, 2026 OOS period. This is the opportunity cost of deploying capital in prediction markets vs. equities. If SPY returned 5% over this window and our strategy returned 7.7% (Extended LR: $77.25 on $1,000), we are beating the benchmark on return but on a fundamentally different risk profile (prediction market binary bets vs. equity market exposure).

**Buy-and-hold BTC** — Total return of spot BTC over the same period. This is the natural crypto benchmark. BTC's return is highly variable and path-dependent; if crypto was in a bull run during our OOS window, a simple long bias would have beaten our model. The comparison contextualizes whether our edge is from model skill or from market direction.

**Key insight:** Our models' positive Sharpe comes from consistent small wins regardless of market direction, not from correctly calling the broad market trend. This is a fundamentally different alpha source than directional crypto exposure.

### Backtest Validity & Limitations

**No look-ahead:** All predictions come from OOS CV folds — the model never trained on the evaluation period.

**Realistic transaction costs:** We model and report TC sensitivity explicitly.

**No survivorship bias:** We include all contracts in the OOS period, not just "interesting" ones.

**Limitations:**
- **Short evaluation window:** 1,596 contracts over 6 weeks. This is enough to estimate performance with reasonable confidence but not enough to characterize performance across different market regimes.
- **Simplified payoff model:** We approximate Kalshi payoffs as ±$1 rather than modeling actual prices. Real entry prices affect PnL substantially.
- **No execution model:** We assume trades execute at the beginning of each contract at the observed opening probability. In practice, there is latency and the price may move before your order fills.
- **Static position sizing:** $1 per trade ignores Kelly criterion or volatility-scaled sizing, which would be more optimal.
- **No regime detection:** The model is not retrained as market conditions change. If market microstructure shifts (e.g., Kalshi gains many more participants and becomes more efficient), model performance will degrade undetected.