"""Reproducible evaluation on the labeled synthetic dataset.

Produces:
  * image-level detection metrics for a NON-AGENTIC baseline (threshold on the
    detector's score) and for the AGENTIC controller (status -> prediction);
  * ROC-AUC on the raw score;
  * pixel-level IoU where ground-truth masks exist;
  * latency, agent iteration counts, and human-review escalation rate.

Run:  python -m src.evaluation.evaluate
All numbers are computed from actual runs; none are hard-coded.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from src.agent.controller import InspectionAgent, Status
from src.evaluation import metrics
from src.vision import detector, preprocessing, synthetic

ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"
BASELINE_THRESHOLD = 0.49  # fixed, documented operating point for the baseline


def _agent_pred(status: Status) -> int:
    """Map agent status to a binary prediction for a fair comparison.

    POTENTIAL_ANOMALY -> 1; NO_ANOMALY -> 0; HUMAN_REVIEW -> abstain (treated as
    'flagged for attention' = 1, since a human-review case is NOT a clean pass).
    """
    return 0 if status == Status.NO_ANOMALY else 1


def run_evaluation(n_normal: int = 40, n_defect: int = 40, seed: int = 7) -> Dict:
    ds = synthetic.make_dataset(n_normal=n_normal, n_defect=n_defect, seed=seed)
    agent = InspectionAgent()

    y_true: List[int] = []
    raw_scores: List[float] = []
    baseline_pred: List[int] = []
    agent_pred: List[int] = []
    iters: List[int] = []
    escalations = 0
    ious: List[float] = []
    det_latency: List[float] = []
    agent_latency: List[float] = []

    for s in ds:
        prep = preprocessing.normalize(s.image)

        t0 = time.perf_counter()
        det = detector.detect(prep.gray, scale=prep.scale)
        det_latency.append(time.perf_counter() - t0)

        t1 = time.perf_counter()
        out = agent.run(prep)
        agent_latency.append(time.perf_counter() - t1)

        y_true.append(s.label)
        raw_scores.append(det.anomaly_score)
        baseline_pred.append(1 if det.anomaly_score >= BASELINE_THRESHOLD else 0)
        agent_pred.append(_agent_pred(out.status))
        iters.append(out.iterations)
        if out.escalated:
            escalations += 1

        # Pixel-level IoU for defect samples that produced a prediction mask.
        if s.label == 1 and det.anomaly_map is not None:
            pred_mask = cv2.resize(
                (det.anomaly_map >= det.threshold).astype(np.uint8),
                (s.mask.shape[1], s.mask.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            iou = metrics.mask_iou(pred_mask, s.mask > 0)
            if not np.isnan(iou):
                ious.append(iou)

    baseline_m = metrics.confusion(y_true, baseline_pred)
    agent_m = metrics.confusion(y_true, agent_pred)
    auc = metrics.roc_auc(y_true, raw_scores)
    best_t, best_f1 = metrics.best_f1_threshold(y_true, raw_scores)

    result = {
        "dataset": {"n_normal": n_normal, "n_defect": n_defect, "seed": seed},
        "opencv_version": cv2.__version__,
        "baseline": {
            "threshold": BASELINE_THRESHOLD,
            **baseline_m.to_dict(),
        },
        "agent": {
            **agent_m.to_dict(),
            "escalation_rate": round(escalations / len(ds), 4),
            "mean_iterations": round(float(np.mean(iters)), 3),
            "max_iterations": int(np.max(iters)),
        },
        "score_roc_auc": None if np.isnan(auc) else round(auc, 4),
        "best_f1_threshold": {"threshold": round(best_t, 4), "f1": round(best_f1, 4)},
        "pixel_iou": {
            "n": len(ious),
            "mean": round(float(np.mean(ious)), 4) if ious else None,
            "median": round(float(np.median(ious)), 4) if ious else None,
        },
        "latency_ms": {
            "detector_mean": round(float(np.mean(det_latency)) * 1000, 2),
            "detector_p95": round(float(np.percentile(det_latency, 95)) * 1000, 2),
            "agent_mean": round(float(np.mean(agent_latency)) * 1000, 2),
            "agent_p95": round(float(np.percentile(agent_latency, 95)) * 1000, 2),
        },
    }
    return result


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    result = run_evaluation()
    out_path = ARTIFACTS / "evaluation_results.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
