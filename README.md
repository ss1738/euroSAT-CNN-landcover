# EuroSAT Land Cover — from coursework CNN to an assured onboard EO stack

This repo began as an MSc *AI for Space* group project (University of Leicester, graded 90% — `coursework/`): a ResNet-18 CNN classifying EuroSAT satellite imagery. It has since been upgraded into a measured research artifact: **an onboard-class EO classifier with layered assurance — calibrated refusal, out-of-distribution monitoring, compression-vs-assurance trade-offs, and radiation-fault behaviour — designed as the perception front-end for a formally verified action-gating kernel** (separate project). Every claim below is measured; `experiments/` + `results/` reproduce all of it (single RTX 5090, EuroSAT RGB, splits/seeds fixed).

Six findings, in the order we learned them:

## 1. Quantization silently breaks conformal guarantees (n=20 seeds)
Split-conformal calibration done on the FP32 model and reused on the INT8 deployment artifact **undercovers at every α tested** (e.g. 88.8±0.8% vs a 90% target; gap grows to 2 points at α=0.15), while INT8 costs almost no accuracy (98.13%→98.03%). Recalibrating on the quantized model restores target coverage exactly. *Qualification rule: calibrate the artifact that flies, not the model you trained.*

## 2. Conformal refusal is blind to sensor noise (n=20)
Refusal (empty prediction set) rises appropriately under blur (49%) and brightness shift (39%) — but under additive Gaussian noise the model becomes **confidently wrong**: accuracy 13%, refusal 6.9% (*below* the clean rate). The guarantee dies with no observable signal.

## 3. Logit-space OOD (energy score) is blind to the same noise (n=5)
Energy-score detection at 5% clean FPR catches blur (45%) and brightness (34%) but only **1.5% of noise** — noise inflates logits, reading as *extra in-distribution*. Two independent learned monitors, same blind spot.

## 4. A near-free input-statistics check closes it completely (n=20)
A high-frequency-energy + channel-std envelope on the raw image (one 3×3 average-pool and two reductions — MCU-trivial) **catches 100% of noise in 20/20 seeds at 1.7% clean cost**. Three-check gate (conformal ∧ energy ∧ input-stats), ResNet18 INT8: clean residual risk **0.32±0.11%** at 88.3% acceptance; noise **0 accepted**; brightness 0.65%. Defense-in-depth, each layer catching what the others miss.

## 5. The blur dial buys exposure, not conditional risk (n=20)
Sweeping the envelope's lower bound flags 12→40% of blurred inputs, but error *among accepted blurred inputs* stays flat (~27–31%) at every operating point: in-envelope blur is genuinely ambiguous. Input-space filtering reduces blur **exposure**; blur **risk** must come from the uncertainty layers. Quantified as a curve, stated as a limitation.

## 6. Compression has an assurance price; bit-flips have no output-side alarm
**Ladder** (n=5): KD students at 99.5 KB / 42.8 KB / **25.4 KB** INT8 hold 94.8 / 92.5 / 89.0% accuracy with residual risk 2.1 / 4.7 / 9.4% — and **noise blocking stays 100% at every size** (the input-space check is model-independent). A 450× shrink with the assurance cost measured, not assumed.
**SEU study** (10 trials/level): single weight bit-flips are fully absorbed; at 100 flips the teacher drops to 55.6% accuracy while the gate still accepts 76.6% of inputs (41.6% of them wrong). **Output-side monitoring cannot detect weight corruption** — memory-integrity protection (weight digests / EDAC) is a separate, necessary layer, which is exactly what the downstream verified kernel's self-integrity check provides.

## Architecture this feeds
```
raw image ─ input-stats envelope ─┐
INT8 classifier ─ conformal sets ─┼─ ACCEPT/REFUSE ─→ formally verified action-gating
              └─ energy score ────┘                   kernel (machine-checked proofs,
                                                      cFS-integrated; separate project)
```

## Also relevant
- Toolchain finding: **QAT gained nothing over PTQ** at ≤100 KB scale (95.75% vs 95.78%) — post-training quantization suffices, simplifying any qualification story.
- Prior art: weighted conformal prediction for onboard EO domain shift exists (DLR, SPIE 2025) — cited as the foundation for the perception layer; the contribution here is the *systems integration, measured failure modes, and qualification framing* (ECSS-E-HB-40-02A), not conformal methodology.
- Next: STM32/WB55 on-silicon latency for the 25–100 KB students beside the kernel's WCET harness; cFS end-to-end demo; real (non-synthetic) sensor-degradation data.

## Repo map
`coursework/` — the original module project (report, slides, notebook, demo video; graded 90%)
`experiments/eurosat_stage1..6.py` — one script per stage, self-contained
`results/*.json` — raw measured outputs (per-seed rows + aggregates) for every table above

*MSc AI for Business Intelligence, University of Leicester. Experiments 2026-08.*
