# Evaluation

All numbers here come from `python -m src.evaluation.evaluate`, which writes
`artifacts/evaluation_results.json`. Charts come from
`python scripts/make_charts.py`. Everything is deterministic (fixed seeds) and
regenerable.

## Protocol (fixed before reading results)

- **Dataset:** balanced synthetic set, 40 normal + 40 defect, seed 7
  (`src/vision/synthetic.py`). Defect kinds cycle scratch / spot / dent.
- **Image-level detection, two systems:**
  - *Baseline:* fixed threshold on the detector's image-level score
    (operating point 0.49).
  - *Agent:* the controller's final status mapped to a binary prediction
    (`no_anomaly` → 0; `potential_anomaly`/`human_review` → 1, because a
    human-review case is **not** a clean pass).
- **Score ranking:** ROC-AUC via the Mann–Whitney rank formulation
  (`src/evaluation/metrics.py`).
- **Localization:** pixel IoU between the thresholded anomaly map and the
  ground-truth mask, over defect samples.
- **Agent behavior:** escalation rate, mean/max iterations.
- **Latency:** per-image wall-clock for detector and full agent.

## Headline results (OpenCV 5.0.0)

| Metric | Baseline | Agent |
|---|---|---|
| Precision | 1.000 | 1.000 |
| Recall | 0.975 | 0.975 |
| F1 | 0.987 | 0.987 |
| Accuracy | 0.988 | 0.988 |
| False positives | 0 | 0 |
| False negatives | 1 | 1 |
| Escalation rate | — | 3.75% |
| Mean / max iterations | — | 2.04 / 3 |

- **ROC-AUC (raw score):** 0.994
- **Best-F1 threshold:** 0.515 (F1 0.987)
- **Pixel IoU:** mean 0.704, median 0.791 (n=40)
- **Latency:** detector mean 4.2 ms / p95 4.7 ms; agent mean 6.0 ms / p95 7.8 ms

## Charts

- `artifacts/score_distribution.png` — normal vs defect score histograms with
  the baseline threshold marked. Normal scores sit ≤ ~0.48; defects cluster
  high (~0.9) with a short low tail (faint scratches).
- `artifacts/pr_curve.png` — precision–recall on the raw score.
- `artifacts/demo_panel.png` — original / annotated / heatmap for a spot defect.

## Threshold calibration (how the heuristics were set)

Observed on the dataset: clean-surface quality metrics — blur-var ≥ ~80
(sharp), brightness ~121–148, contrast std ~7–13, no clipping. Detector
scores: normal ≤ 0.48, defect in [0.43, 0.97]. From these:

- quality flags calibrated to the controlled surface
  (`CONTRAST_STD_MIN=4.0`, `BLUR_VAR_MIN=25.0`);
- agent band `SCORE_LOW=0.50`, `SCORE_HIGH=0.55`, placing faint scratches in
  the grey zone where re-inspection fires.

These are **documented heuristics tuned to the controlled-surface assumption,
not calibrated probabilities**, and must be re-tuned for a different product.

## Failure case (documented honestly)

One defect — a very faint, thin scratch whose contrast approaches the surface
grain — scored below the clean cutoff and was reported as **no anomaly**: a
genuine **false negative**. Three other faint scratches scored in the grey zone
(0.45–0.50); the agent re-inspected, could not corroborate, and routed them to
**human review** rather than guessing. This is the intended safety behavior and
the clearest illustration of the agent's value over a naive threshold.

## Known limitation of this evaluation

The dataset is **synthetic** with deliberately separable defects under
controlled lighting. Metrics are an **upper-bound sanity check**, not a
real-world benchmark. Real-world performance on subtle, low-contrast, or
lighting-variable defects will be lower and must be measured on a real labeled
dataset (e.g. MVTec AD) before any operational claim.
