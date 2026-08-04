# Safety-Cage Evidence Package for an Assured Onboard EO Classifier
### A worked example oriented to ECSS-E-HB-40-02A (ML qualification handbook)

**Status:** public worked example, v0.1, August 2026. To our knowledge the first
public end-to-end attempt at an evidence package for the handbook's safety-cage
architecture since its publication (15 Nov 2024). Clause-level traceability to
the handbook text is marked [MAP] where section-level mapping is asserted from
the handbook's published structure and needs verification against the full text
before use in a formal data package. Every measurement cited is reproducible
from `experiments/` and `results/` in this repository.

---

## 1. System under assurance and operational domain

- **Function:** land-cover classification of 64x64 optical tiles (10 classes,
  EuroSAT distribution) producing per-tile class decisions consumed by an
  onboard autonomy function (retasking / downlink prioritisation).
- **Deployment artifact:** ResNet18-class CNN, 8-bit weight quantization
  (post-training, per-channel); MCU-class distilled variants measured down to
  25.4 KB. Artifact identity is bound by sha256 weight digest (see Assurance
  Card, section 6).
- **Criticality assumption:** the classifier itself is never the final
  authority; it sits behind a safety cage (section 3). This is the handbook's
  intended pattern for category B/C/D ML functions. [MAP]
- **Operational domain declaration:** clean-sensor optical tiles statistically
  consistent with the calibration set. Departure from this domain is detected
  at runtime by the cage's input-statistics envelope (section 3, layer 3)
  rather than assumed away.

## 2. Model development and data evidence

| Evidence item | Where |
|---|---|
| Training procedure, seeds, splits (70/15/15) fully scripted | `experiments/eurosat_stage1..12.py` |
| Multi-seed variance reported for every headline number (n=20 where claimed) | `results/*.json`, README findings 1-12 |
| Compression procedure (PTQ) and its measured side effects | Findings 1, 8, 9, 10, 11 |
| No hand-tuning on test data: calibration/test separation enforced in every script | scripts |

## 3. The safety cage: architecture and measured behaviour

Three input-side checks, one output-side guarantee, one action-side gate:

```
raw tile -- [L3] input-statistics envelope --+
model    -- [L1] conformal prediction sets --+--> ACCEPT / REFUSE --> verified
         -- [L2] energy (logit) envelope  ---+                       action-gating
                                                                     kernel
```

- **L1 conformal (guarantee layer):** split conformal, alpha = 0.10. Measured
  coverage on target after recalibration on the deployed artifact:
  90.0 +/- 0.7% (n=20). Refuses by empty prediction set.
- **L2 energy envelope:** catches distribution shift the conformal layer
  misses (45% of blur, 34% of brightness shift at 5% clean false-positive
  budget). Measured blind to additive sensor noise (1.5% detection) - which is
  why L3 exists. (Findings 3, 4.)
- **L3 input-statistics envelope:** high-frequency energy + channel standard
  deviation of the raw tile against a calibrated envelope. Catches 100% of
  noise-corrupted inputs in 20/20 seeds at 1.7% clean cost. Model-independent:
  detection performance is unchanged across every model size tested.
  (Findings 4, 6.)
- **Action gate:** accepted decisions pass to a formally verified action-gating
  kernel (separate component: no_std Rust, 22 machine-checked proofs including
  the C ABI, NASA cFS integration, WCET-instrumented). The kernel enforces
  flight-rule invariants on the actions themselves and fails closed.
- **Measured end-to-end residual risk** (error rate among ACCEPTED inputs,
  teacher artifact, n=20): clean 0.32 +/- 0.11%, brightness 0.65%, sensor
  noise 0 (nothing accepted), blur 31% at 40% acceptance - declared as the
  cage's known weak axis, with the operating-point trade-off curve published
  (Finding 5), not hidden.

## 4. Compression qualification evidence (the novel part)

The handbook must be satisfied for the artifact that flies, not the model that
trained. Three measured results make that tractable:

1. **Calibrate the artifact, not the ancestor.** Reusing FP32 calibration on
   the quantized artifact undercovers at every alpha tested (Finding 1, n=20).
   Recalibration on the deployed weights restores target coverage exactly.
2. **Coverage loss is forecastable before deployment.** The coverage deficit
   under stale calibration is upper-bounded by the TV distance between
   reference and deployment score distributions on the labeled calibration
   set; bound held in 60/60 runs across 2 datasets x 2 architectures x 5
   bit-widths, near-tight on peaked (mission-like) score distributions
   (Findings 9, 10). This converts an unquantified qualification risk into a
   number computable on the ground.
3. **Informativeness has a precision floor.** Mean prediction-set size versus
   bit-width gives a one-line acceptance test for how far the artifact may be
   compressed (Finding 11: knee at 4-bit for ResNet-class, one bit earlier
   for MobileNet-class).

## 5. Fault and radiation evidence

- **Output-side monitoring cannot detect weight corruption:** 83-100% of
  single stored-weight bit flips are output-silent across all storage formats
  tested (Finding 12). Any monitoring argument built on output observation
  alone is therefore insufficient - stated as a negative result with data.
- **Integer weight storage is intrinsic fault containment:** 100 stored-weight
  flips at 8-bit produced no measurable accuracy change (98.2% -> 98.2%),
  while 100 FP32 flips halved accuracy with the gate blind (Finding 12).
  Storage-format choice is a fault-tolerance decision and is declared in the
  Assurance Card.
- **Detection mechanism with measured cost:** periodic sha256 verification of
  the weight region; measured at 4.3x one CPU inference, i.e. ~4% duty
  overhead at one hash per 100 frames. Software fault injection only; not a
  radiation test - beam characterisation is future work and stated as such.

## 6. Operations: calibration lifetime and artifact identity

- **Identity:** every deployed artifact is bound to its sha256 weight digest
  via its Assurance Card (spec AC-0.1, github.com/ss1738/qualkit). The card
  carries the guarantee, its validity assumptions, the compression forecast,
  efficiency verdicts, and fault exposure - generated from executed checks,
  never hand-written.
- **Calibration drift over mission lifetime:** the maintenance action for the
  guarantee is a conformal score refresh (kilobytes of labeled scores), not
  model retraining. Measured under a simulated 24-week progressive sensor
  degradation profile in `results/stage12_results.json` (drift study;
  refresh-cadence versus coverage curves).
- **Enforcement in the development loop:** the same checks gate model changes
  in CI (github.com/ss1738/qualkit-action): a pull request that degrades the
  coverage forecast or balloons set size fails the build. Qualification
  becomes an enforced process property rather than a one-time document.

## 7. Residual risks and honest limitations

1. Blur-type degradation: exposure reducible, conditional risk not (Finding
   5). Declared operating limitation; mitigation options quantified.
2. Corruptions in evidence are synthetic (noise/blur/brightness transforms);
   real degraded-sensor campaigns are future work.
3. Exchangeability is assumed within the declared domain; the cage detects,
   but does not certify against, arbitrary unknown-unknown shifts.
4. Software fault injection, not beam testing (section 5).
5. Clause-level handbook traceability pending verification against the full
   ECSS-E-HB-40-02A text [MAP items].
6. The verified kernel's proofs cover its implementation properties; the
   toolchain (rustc, Kani) is not itself qualified - a known, industry-wide
   open item (Ferrocene qualification is the tracked path).

## 8. Traceability summary

| Handbook theme [MAP] | Evidence in this package |
|---|---|
| ML model development and data quality | Section 2; scripts + multi-seed results |
| Safety-cage architecture for cat. B/C/D | Section 3; measured layer-by-layer |
| V&V of the deployed (compressed) artifact | Section 4; Findings 1, 8-11 |
| Robustness / fault tolerance | Section 5; Finding 12 |
| Operations, monitoring, maintenance | Section 6; drift study, cards, CI |
| Limitations and residual risk declaration | Section 7 |

---

*Prepared from the measured findings in this repository. Contact: Satyawan
Singh. The companion tooling (qualkit, qualkit-action, Assurance Cards) is
open source; evidence-package preparation against specific mission data is
available as a service.*
