# Planned paper: claim-to-evidence map

Working title: "Forecasting conformal coverage loss under model compression."
Target: arXiv preprint, then venue. Each claim below cites its finding and raw
data; nothing enters the paper without a row here.

| # | Claim | Evidence |
|---|---|---|
| C1 | Stale calibration under PTQ undercovers systematically, growing with alpha | Finding 1 (n=20) |
| C2 | Coverage deficit is upper-bounded by TV distance of score shift (Barber-style bound, instantiated for quantization) | Finding 9 |
| C3 | The bound is near-tight on peaked score distributions, conservative otherwise | Findings 9, 10 |
| C4 | Holds 60/60 across 2 datasets x 2 architectures x 5 bit-widths | Finding 10 |
| C5 | Recalibration restores validity at every precision; efficiency has a knee | Findings 8, 11 |
| C6 | Deployment corollary: score refresh maintains validity under drift; set size is the model-health signal | Finding 13 |

Mandatory pre-submission checks: read DLR SPIE 13670 (onboard CP) in full;
re-verify TV estimator bias at 3-bit (the 0.001 nominal violation).
