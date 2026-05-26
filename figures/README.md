# figures/

Every figure here is used in `docs/presentation/` (gitignored). All deck figures are rendered in Times New Roman with text ≥ 32 pt (titles 36–38 pt) for 60-person-room legibility.

## Layout

```
figures/
├── eda/                                # data + calibration figures
│   ├── data_joint_panel.png            slide 6
│   ├── probability_convergence.png     slide 7
│   ├── hourly_accuracy.png             slide 8
│   ├── accuracy_heatmap.png            slide 9
│   ├── methods_diagram.png             slide 10
│   ├── calibration_panel.png           appendix slide 19
│   └── conviction_spread_accuracy.png  appendix slide 20
├── logReg/                             # incremental-information test + LR appendix
│   ├── incremental_information.png     slide 13
│   ├── feature_importance.png          appendix slide 21
│   ├── decision_boundary.png           appendix slide 22
│   └── volatility_forecast.png         appendix: HAR-RV vs GARCH vs Kalshi
├── lead_lag.png                        slide 11
├── stress_events_lead_lag.png          appendix: lead-lag by RV regime
└── variance_ratio.png                  slide 12
```

To regenerate any figure, see the [quickstart](../README.md#quickstart). Each script writes to the path above.
