---
output:
  pdf_document: default
  html_document: default
---
# Speaker Notes — OQG Presentation (Zoe's Section)

Reference deck: `presentation/presentation_d4.pptx`. My section runs from slide 9 to the end. Total budget about 4 to 5 minutes. Format follows my thesis speaker notes: short bullets, indented sub-bullets, spoken aloud register.

---

## Slide 9: Testing the Signal Against the Underlying  *(transition)*
**Zoe takes over from Xavi, ~25 sec**

- Xavi's section has characterized the within-Kalshi signal in the BTC contract panel
- The question I take up is whether Kalshi P(UP) carries directional information beyond what is already reflected in the contemporaneous spot price
    - every Kalshi BTC contract resolves on Coinbase spot, so spot is the natural benchmark
    - for Kalshi to be useful for trading the underlying asset, it must carry information incremental to the spot price process
- I present three diagnostics, ordered so that each is motivated by the result of the preceding one
    - lead-lag cross-correlogram
    - variance ratio test
    - incremental information test

---

## Slide 10: Spot Leads P(UP) by Two Minutes
**~45 sec**

- The first diagnostic addresses a timing question
    - at minute frequency, does P(UP) lead or lag the spot return?
    - under the hypothesis that Kalshi acts as a leading sentiment signal, P(UP) should update before the underlying; under the alternative that Kalshi is downstream of spot, it should update after

Method
- Pearson cross-correlation across time lags
    - correlate the within-contract first difference ΔP(UP)_t with the spot 1-minute log return r_{t+k}
    - sweep k ∈ [−10, +10] minutes
- Sign convention: k > 0 corresponds to Kalshi leading spot; k < 0 corresponds to spot leading Kalshi
- Cluster bootstrap 95% confidence intervals over contracts (Cameron, Gelbach and Miller, 2008) to account for within-contract serial dependence

Result
- Kalshi lags spot by approximately 2 minutes
    - the cross-correlogram peaks at k = −2 with sample correlation r = +0.51 for BTC
    - all lags at which Kalshi could lead spot (k ≥ 0) are statistically indistinguishable from zero or negative
- A component of the observed lag is mechanical
    - Coinbase 1-minute candle aggregation contributes approximately 0.5 minute
    - the Kalshi scraper averages mid-quotes over the prior minute, contributing approximately another 0.5 minute
    - the substantive reaction component is bounded below at no less than 1 full minute

---

## Slide 11: P(UP) Deviates from the Martingale Null
**~55 sec**

- The cross-correlogram result motivates a complementary test that does not depend on the join between two distinct data panels
    - I test whether the Kalshi P(UP) series, taken alone, is consistent with a random walk
- Rationale for the random-walk null
    - the random walk is the canonical benchmark for weak-form market efficiency (Fama, 1970)
    - under weak-form efficiency, the current price reflects all available information, so past returns carry no predictive structure for future returns
    - liquid spot markets, BTC included, generally fail to reject this null at minute frequency
    - rejection of the null indicates slow information absorption — i.e., exploitable serial correlation in past returns

Method
- Lo and MacKinlay (1988) variance ratio test
- Under the random-walk null, the variance of t-step returns scales linearly with t
    - equivalently, the q-step return variance equals q times the 1-step return variance
    - so VR(q) ≡ Var(q-step) / [q · Var(1-step)] = 1 at every horizon q
- VR(q) > 1 indicates positive serial correlation, consistent with under-reaction (momentum)
- VR(q) < 1 indicates negative serial correlation, consistent with mean reversion
- The Hurst exponent H summarizes the scaling law in a single parameter
    - the law VR(q) = q^(2H − 1) implies a slope of 2H − 1 in log VR(q) vs log q
    - H = 0.5 corresponds to the random-walk null; H > 0.5 indicates persistence (under-reaction); H < 0.5 indicates anti-persistence (mean reversion)
- I evaluate at q ∈ {2, 3, 5, 7, 10} minutes
    - the range covers the 2-minute lag identified in the preceding test and remains within the 15-minute contract window
    - five horizons provide sufficient leverage to estimate the scaling slope robustly

Result
- Spot does not reject the null
    - VR(q) lies within 1% of 1.00 across all horizons
    - Hurst H = 0.498, 95% CI contains 0.5
- Kalshi P(UP) rejects the null
    - VR(2) = 1.24,  VR(10) = 1.41
    - Hurst H = 0.537, 95% CI excludes 0.5

Interpretation
- Rejection of the random-walk null implies that P(UP) is predictable from its own past
    - VR(q) > 1 specifically indicates that following a move in P(UP), additional same-direction moves are expected over the subsequent several minutes
    - this is the signature of under-reaction: P(UP) does not fully incorporate new information in a single adjustment, but converges to the updated level gradually as additional traders observe and respond
- The finding is consistent with the lead-lag result
    - spot moves first; Kalshi is downstream and adjusts gradually
    - the gradual adjustment manifests as positive autocorrelation in the Kalshi P(UP) series
- Two timing diagnostics, from independent angles, yield consistent conclusions
    - spot leads Kalshi by approximately 2 minutes (cross-series test)
    - Kalshi P(UP) exhibits significant under-reaction at 2 to 10 minute horizons (within-series test)

---

## Slide 12: Kalshi Features Add No Content Above Spot
**~50 sec**

- The first two diagnostics characterize how Kalshi *behaves*: it lags spot by 2 minutes and under-reacts to its own moves
- The next question is predictive rather than descriptive: does the 2-minute lag impair Kalshi's ability to contribute directional information when used as a feature set alongside spot?
    - under the hypothesis that Kalshi carries information orthogonal to spot, augmenting a spot-only model with Kalshi features should improve out-of-sample performance
    - under the alternative that Kalshi is fully subsumed by the spot price process, the combined model should perform at parity with, or worse than, spot-only — since additional features contribute estimation variance without bias reduction in finite samples

Method
- Head-to-head classification on the same out-of-sample contracts using three nested feature sets
    - Spot-only: 8 features derived from BTC spot prices in the first 5 minutes of the contract window (returns, realized volatility, log volume)
    - Kalshi-only: 9 features derived from the Kalshi panel
        - opening P(UP), conviction spread, momentum at 1 and 3 minutes, mean and standard deviation of P(UP) over minutes 0–4, log volume, hour of open
    - Combined: union of both feature sets, 17 features
- 5-fold time-series cross-validation on 1,595 out-of-sample contracts; training on past, testing on future, with random shuffling precluded by the time-series structure
- DeLong, DeLong and Clarke-Pearson (1988) paired non-parametric test for AUC comparisons on shared evaluation data
- Bootstrap 95% confidence intervals on out-of-sample AUC

Result
- Spot-only AUC = 0.796, 95% CI [0.77, 0.82], 73.1% accuracy
- Kalshi-only AUC = 0.690, 64.6% accuracy
- Combined AUC = 0.758, 71.2% accuracy — strictly worse than spot-only
- DeLong p < 0.001 for both the Kalshi-vs-Spot and Combined-vs-Spot gaps

Interpretation
- The combined model performing strictly worse than spot-only is the diagnostic signature of an irrelevant feature block
    - in finite samples, augmenting the feature set with non-orthogonal predictors contributes estimation variance without bias reduction
    - this is consistent with the spot price process already encoding the directional information; Kalshi features contribute noise to the in-sample fit without contributing signal to the out-of-sample prediction

---

## Slide 13: Limitations & Next Steps
**~60 sec**

- This research provides a characterization of Kalshi as an alternative data source for investor sentiment
- Kalshi P(UP) on BTC is found to be:
    - Calibrated — probabilities are backed by real-money trading and meaningfully track realized outcomes, unlike text-based sentiment proxies that yield only an ordering
    - Lagged behind contemporaneous spot prices by approximately 2 minutes
    - Exhibits under-reaction to its own information at 2 to 10 minute horizons
    - Adds no incremental value above spot for short-horizon directional prediction

Limitations
- Identification
    - the 2-minute lag confounds three sources
        - Coinbase 1-minute candle aggregation
        - Kalshi scraper averaging of mid-quotes over the prior minute
        - substantive reaction at Kalshi — which itself conflates belief-updating delay with Kalshi microstructure (wider bid-ask spreads, thinner order book, fewer quote updates per minute)
    - future work using tick-level spot data together with Kalshi order book data could identify the contribution of each component
- Inference
    - the cluster bootstrap by contract preserves within-contract dependence but treats contracts as independent draws across calendar time, despite shared macro state across adjacent contracts
    - future work could re-estimate confidence intervals using a calendar-time block bootstrap; qualitative findings should hold, magnitudes may tighten
- Generalizability
    - six weeks of moderate-volatility data represents a single market regime
    - the 2-minute lag and the 0.10 AUC gap should not be treated as structural parameters until they replicate in alternative regimes
    - future work could expand the sample range or evaluate during stress events (e.g., CPI releases, BTC-specific news shocks)

The most promising next step concerns the second moment: does Kalshi forecast *realized volatility* even when it does not forecast direction?
- the conviction spread |P(UP) − 0.5| serves as a direct measure of crowd uncertainty — narrow when trader consensus is high, wide when consensus is low — and crowd uncertainty is conceptually aligned with what volatility quantifies
- the test would benchmark conviction spread against HAR-RV (Corsi, 2009) and GARCH, both of which forecast volatility from past returns rather than market-derived sentiment
- if Kalshi contributes incremental information above those benchmarks, this is the most promising practical use of the data source — risk management, options pricing, position sizing

---

## Slide 15: Questions
**~15 sec**

- Three takeaways about Kalshi as a data source
    - Calibration: unlike text-based sentiment proxies, P(UP) provides calibrated probability estimates rather than mere orderings
    - Speed: the 2-minute lag and the positive autocorrelation in P(UP) quantify the speed at which a real-money crowd absorbs price information at minute frequency
    - Scope: the directional channel is already reflected in spot prices; the remaining use cases lie in second-moment forecasting and calibrated minute-frequency sentiment measurement, not in directional alpha for the underlying asset
- Happy to take questions

---

## Backup Q&A

**Q: Why join Kalshi to spot by exact UTC minute rather than nearest timestamp?**
- Exact UTC minute is conservative
    - the scraper already averages mid-quotes over the prior minute
    - so the Kalshi value at minute t is the average over (t-1, t]
    - and the Coinbase candle at minute t is also (t-1, t]
- Nearest-timestamp matching would smear the join window and bias the contemporaneous correlation upward
    - artifactually making Kalshi look more contemporaneous with spot than it is

**Q: Why cluster bootstrap by contract rather than by day or hour?**
- Within-contract observations are mechanically dependent
    - P(UP) at minute t in a contract is conditioned on P(UP) at minute t-1 in the same contract
- Contract is the smallest unit that is plausibly exchangeable across calendar time
    - day or hour clustering would be more conservative but coarser
    - I note this in the limitations slide

**Q: Why 5-min feature window specifically? Why not 3, or 7?**
- Bias-variance tradeoff
    - shorter window leaves more forward window to predict on, but features are noisier
    - longer window gives cleaner features but the contract has already partly converged
- 5 minutes balances both
    - the convergence curve Xavi showed is still in the early plateau at t=5
    - and the 10 remaining minutes is the prediction window that matters for trading

**Q: Could the result reverse during high-volatility regimes?**
- Possible and worth testing
    - Kalshi may be slower to update during stress (wider spreads, thinner liquidity)
    - which would widen the lag, not narrow it
- But it's also possible that informed traders concentrate on Kalshi during news events
    - in which case Kalshi could lead spot transiently
- Six weeks of moderate vol is one regime; this is the generalizability point on slide 13

**Q: Why no XGBoost or nonlinear model in the head-to-head?**
- The question is about feature information, not model capacity
    - a regularized linear model already picks up additive contributions; if Kalshi added orthogonal directional signal, the combined LR should at least match spot
- Our finding is combined < spot, with DeLong p < 0.001
    - I treat the linear test as a lower bound on incremental information at this sample size
    - a nonlinear model could in principle find Kalshi-specific interactions the linear model misses, which would shift the magnitude
    - but for the result to flip (combined > spot) the nonlinear gain would need to be large enough to overcome the 0.796 spot baseline — implausible given the linear test finds essentially zero incremental signal
- happy to run XGBoost as a robustness check; my expectation is it doesn't change the qualitative answer

**Q: Could there be alpha within Kalshi itself, even if Kalshi adds no value for trading BTC spot?**
- Yes, and that's a separate research question
    - the variance ratio result (H = 0.537, VR(2) = 1.24) says P(UP) under-reacts to its own past moves
    - that's a predictable pattern in Kalshi's own series — exploitable in principle
- Practical bar is high though
    - 15-min binary contracts have non-trivial spread and fee costs relative to the size of the predictable move
    - and an exploitable serial correlation in P(UP) at minute frequency may not survive once you account for execution
- More important framing: Kalshi-internal alpha is testing market efficiency at Kalshi
    - my question was whether Kalshi adds value for trading the underlying asset
    - those are two different papers
