# data/

Raw and derived datasets.

```
data/
├── btc/         primary inputs for the headline analysis
│   ├── kalshi_btc_prices.csv      Kalshi 1-min P(UP)/P(DOWN), 2,264 BTC 15-min contracts
│   ├── spot_btc_1m.csv            Coinbase BTC-USD 1-min OHLCV bars
│   └── btc_contracts.csv          one row per contract with engineered features
├── eth/         retained for earlier cross-asset work; not used in the headline
│   ├── kalshi_eth_prices.csv
│   ├── spot_eth_1m.csv
│   └── eth_contracts.csv
└── cleaned/     analysis outputs
    ├── merged_contracts.csv       BTC × ETH contracts joined by exact UTC minute (1,916 rows)
    ├── pm_spot_lead_lag.json      lead-lag diagnostic result
    ├── variance_ratio.json        VR test result
    ├── incremental_summary.json   incremental-information AUCs + DeLong p-values
    └── incremental_oos.csv        per-contract OOS predicted P(UP) per model
```

Window: 15 Feb – 28 Mar 2026 (six weeks, moderate volatility). One scraper outage of 13 days mid-window (shaded on slide 6).

To regenerate the `cleaned/` files from raw inputs, see the [project quickstart](../README.md#quickstart).
