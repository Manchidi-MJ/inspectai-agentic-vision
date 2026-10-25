"""Metric helpers implemented without scikit-learn (keeps deps light).

All metrics are computed from actual predictions. Nothing is hard-coded.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple


@dataclass
class ClassificationMetrics:
    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else 0.0

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def accuracy(self) -> float:
        d = self.tp + self.fp + self.tn + self.fn
        return (self.tp + self.tn) / d if d else 0.0

    def to_dict(self) -> dict:
        return {
            "tp": self.tp, "fp": self.fp, "tn": self.tn, "fn": self.fn,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "accuracy": round(self.accuracy, 4),
        }


def confusion(y_true: Sequence[int], y_pred: Sequence[int]) -> ClassificationMetrics:
    tp = fp = tn = fn = 0
    for t, p in zip(y_true, y_pred):
        if t == 1 and p == 1:
            tp += 1
        elif t == 0 and p == 1:
            fp += 1
        elif t == 0 and p == 0:
            tn += 1
        else:
            fn += 1
    return ClassificationMetrics(tp=tp, fp=fp, tn=tn, fn=fn)


def roc_auc(y_true: Sequence[int], scores: Sequence[float]) -> float:
    """ROC-AUC via the Mann-Whitney U / rank formulation.

    Equivalent to the probability that a random positive outranks a random
    negative, with ties counted as 0.5.
    """
    pos = [s for t, s in zip(y_true, scores) if t == 1]
    neg = [s for t, s in zip(y_true, scores) if t == 0]
    if not pos or not neg:
        return float("nan")
    wins = ties = 0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1
            elif p == n:
                ties += 1
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def pr_curve(y_true: Sequence[int], scores: Sequence[float]) -> Tuple[List[float], List[float], List[float]]:
    """Return (thresholds, precision, recall) sweeping unique score thresholds."""
    thresholds = sorted(set(scores), reverse=True)
    precisions: List[float] = []
    recalls: List[float] = []
    for t in thresholds:
        preds = [1 if s >= t else 0 for s in scores]
        m = confusion(y_true, preds)
        precisions.append(m.precision)
        recalls.append(m.recall)
    return thresholds, precisions, recalls


def best_f1_threshold(y_true: Sequence[int], scores: Sequence[float]) -> Tuple[float, float]:
    """Return (threshold, f1) maximizing F1 over observed score thresholds."""
    thresholds = sorted(set(scores))
    best_t, best_f1 = 0.0, -1.0
    for t in thresholds:
        preds = [1 if s >= t else 0 for s in scores]
        f1 = confusion(y_true, preds).f1
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


def mask_iou(pred_mask, true_mask) -> float:
    """Intersection-over-union between two binary masks (uint8/bool)."""
    import numpy as np
    p = pred_mask.astype(bool)
    t = true_mask.astype(bool)
    inter = int((p & t).sum())
    union = int((p | t).sum())
    return inter / union if union else float("nan")
