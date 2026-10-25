"""Generate evaluation charts from an actual evaluation run.

Produces:
  artifacts/score_distribution.png  (normal vs defect score histograms)
  artifacts/pr_curve.png            (precision-recall curve on raw score)
  artifacts/demo_panel.png          (original | annotated | heatmap for a defect)
"""
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation import metrics  # noqa: E402
from src.vision import detector, preprocessing, synthetic  # noqa: E402

ART = ROOT / "artifacts"
ART.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ds = synthetic.make_dataset(n_normal=40, n_defect=40)
    y, scores = [], []
    for s in ds:
        prep = preprocessing.normalize(s.image)
        r = detector.detect(prep.gray, scale=prep.scale)
        y.append(s.label)
        scores.append(r.anomaly_score)
    y = np.array(y)
    scores = np.array(scores)

    # 1) Score distribution.
    plt.figure(figsize=(6, 4))
    plt.hist(scores[y == 0], bins=20, alpha=0.7, label="normal")
    plt.hist(scores[y == 1], bins=20, alpha=0.7, label="defect")
    plt.axvline(0.49, color="k", ls="--", label="baseline threshold 0.49")
    plt.xlabel("image-level anomaly score")
    plt.ylabel("count")
    plt.title("Anomaly score distribution (AUC=%.3f)" % metrics.roc_auc(y, scores))
    plt.legend()
    plt.tight_layout()
    plt.savefig(ART / "score_distribution.png", dpi=120)
    plt.close()

    # 2) PR curve.
    thr, prec, rec = metrics.pr_curve(y, scores)
    plt.figure(figsize=(6, 4))
    plt.plot(rec, prec, marker=".")
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.title("Precision-Recall (raw score)")
    plt.ylim(0, 1.05)
    plt.xlim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(ART / "pr_curve.png", dpi=120)
    plt.close()

    # 3) Demo panel on a clear defect.
    s = synthetic.make_sample(303, defect="spot")
    prep = preprocessing.normalize(s.image)
    r = detector.detect(prep.gray, scale=prep.scale)
    annotated = detector.annotate(s.image, r)
    heat = detector.heatmap_overlay(s.image, r)
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    for a, img, title in zip(
        ax,
        [s.image, annotated, heat],
        ["original", "annotated", "anomaly heatmap"],
    ):
        a.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        a.set_title(title)
        a.axis("off")
    plt.tight_layout()
    plt.savefig(ART / "demo_panel.png", dpi=120)
    plt.close()

    print("wrote charts to artifacts/: score_distribution.png, pr_curve.png, demo_panel.png")


if __name__ == "__main__":
    main()
