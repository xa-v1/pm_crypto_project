# Data

All raw and derived data used by the analyses.

## Layout

```
data/
├── btc/                            # primary data for the headline analysis
│   ├── kalshi_btc_prices.csv       # Kalshi 1-min P(UP)/P(DOWN) panel, 2,264 BTC contracts, Feb–Mar 2026
│   ├── btc_contracts.csv           # one row per BTC contract with engineered features
│   └── spot_btc_1m.csv             # Coinbase BTC-USD 1-min OHLCV, 58,948 bars, same window
├── eth/                            # retained for reproducibility of earlier cross-asset work
│   ├── kalshi_eth_prices.csv
│   ├── eth_contracts.csv
│   └── spot_eth_1m.csv
└── cleaned/                        # outputs of the analyses
    ├── merged_contracts.csv        # Kalshi × Coinbase, 1,916 matched contracts, joined by exact UTC minute
    ├── pm_spot_lead_lag.json       # lead-lag diagnostic result (slide 10)
    ├── variance_ratio.json         # VR test result (slide 11)
    ├── volatility_leadership.json  # appendix: second-moment lead-lag
    ├── cross_asset_lead_lag.json   # appendix: cross-asset lead-lag
    ├── incremental_summary.json    # incremental information test summary (slide 12)
    ├── incremental_oos.csv         # per-contract OOS predicted probabilities
    ├── lead_lag_result.json        # older BTC vs ETH lead-lag (appendix)
    └── lr_*_oos.csv                # OOS predictions from earlier logistic regression scope
```

## Sources

- **Kalshi** — Public Kalshi trade API. 10-second mid-quote samples averaged into 1-minute buckets per contract.
- **Coinbase** — Public Coinbase Exchange API. 1-minute OHLCV candles.
- **Window** — 15 February to 28 March 2026. Six weeks of moderate-volatility conditions; a single regime.

## Reproducing the cleaned outputs

```bash
# spot data (idempotent)
python scripts/kalshi/analysis/eda/pull_coinbase_spot.py

# diagnostics  →  cleaned/<diagnostic>.json
python -m scripts.kalshi.analysis.lagCorr.pm_spot_lead_lag
python -m scripts.kalshi.analysis.varRatio.variance_ratio
python -m scripts.kalshi.analysis.logReg.incremental_information
```
