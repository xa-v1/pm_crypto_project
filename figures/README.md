# figures/

Every figure here appears in the OQG banquet deck (`Presentation-OQG-Banquet.pdf` at the repo root). All deck figures are rendered in Times New Roman with text ≥ 32 pt (titles 36–38 pt) for 60-person-room legibility.

## Layout

```
figures/
├── eda/                                # data + calibration figures
│   ├── data_joint_panel.png            slide 4   (Spot Price Data)
│   ├── probability_convergence.png     slide 5   (Probability Convergence Within Contracts)
│   ├── hourly_accuracy.png             slide 6   (Intraday Variability in Realized Accuracy)
│   ├── accuracy_heatmap.png            slide 7   (Prediction Accuracy by Opening Conviction)
│   ├── methods_diagram.png             slide 8   (Testing the Signal Against the Underlying)
│   ├── calibration_panel.png           appendix  (Calibration)
│   └── conviction_spread_accuracy.png  appendix  (Market Disagreement Signal)
├── logReg/                             # incremental-information test + LR appendix
│   ├── incremental_information.png     slide 14  (Does Kalshi Add Information Above Spot?)
│   ├── feature_importance.png          appendix  (LR Feature Importance)
│   └── volatility_forecast.png         supplementary  (next-steps probe from slide 15: HAR-RV vs GARCH vs Kalshi)
├── lead_lag.png                        slide 10  (Kalshi P(UP) vs BTC Spot Returns at Lag k)
├── stress_events_lead_lag.png          supplementary  (lead-lag by RV regime, addresses limitation on slide 15)
└── variance_ratio.png                  slide 12  (Variance Ratio Test for Efficient Pricing)
```

To regenerate any figure, see the [quickstart](../README.md#quickstart). Each script writes to the path above.
