# Evidence Map

Every finding, its script, its raw data, and its figures. All results land in
`EA_DIR` (default `~/eurosat-assured`); the copies in `results/` are the runs
behind the README numbers.

| Finding | Claim (short) | Script | Raw results | Figures |
|---|---|---|---|---|
| 1 | Quantization breaks conformal coverage; recalibration restores | stage2 | results/sweep_results.json | coverage.png |
| 2 | Conformal refusal blind to sensor noise | stage2 | results/sweep_results.json | corruptions.png |
| 3 | Energy-score OOD blind to the same noise | stage3 | results/stage3_results.json | corruptions.png |
| 4 | Input-statistics check closes the gap (100%, n=20) | stage4, stage5 | results/stage4_results.json, stage5_results.json | gate.png, corruptions.png |
| 5 | Blur threshold reduces exposure, not conditional risk | stage5 | results/stage5_results.json | gate.png |
| 6 | Compression ladder + SEU pilot | stage6 | results/stage6_results.json | ladder_seu.png |
| 7 | Multispectral transfer + bands-per-byte curve | stage7 | results/stage7_results.json | band_curve.png, bands.png |
| 8 | Coverage-vs-precision scaling law | stage8 | results/stage8_results.json | scaling_law.png |
| 9 | Coverage deficit equals TV distance (tight bound) | stage9 | results/stage9_results.json | tv_bound.png |
| 10 | Bound generalises 60/60 across 2x2 grid | stage10 | results/stage10_results.json | generality.png |
| 11 | Efficiency knee (set size vs precision) | stage10 (analysis) | results/stage10_results.json | efficiency.png |
| 12 | Int storage = SEU containment; flips output-silent | stage11 | results/stage11_results.json | seu_factorial.png |
| 1 (INT8 detail) | Fresh-clone verified stage-1 run | stage1 | results/results.json | coverage.png |
| 13 | Drift: refresh the guarantee, not the model | stage12 | results/stage12_results.json | drift.png |

| Paper Tables 2-3 | Baselines, estimators, APS | stage13 | results/stage13_results.json | paper figures |

Companion artifacts: EVIDENCE_PACKAGE.md (qualification framing) ·
[qualkit](https://github.com/ss1738/qualkit) ·
[qualkit-action](https://github.com/ss1738/qualkit-action) ·
[live telemetry](https://ss1738.github.io/gate-telemetry/)
