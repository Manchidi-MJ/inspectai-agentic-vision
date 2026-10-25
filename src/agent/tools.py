"""Tools the agent can invoke. Each is a thin, pure wrapper over the vision
engine so the controller composes behavior from real analysis, not canned text.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from src.vision import detector, image_quality
from src.vision.preprocessing import Prepared


def assess_image_quality(prep: Prepared) -> Dict:
    """Tool: measure image quality. Returns the raw quality report."""
    return image_quality.assess(prep.gray).to_dict()


def analyze_image(prep: Prepared, thresh_k: float = detector.THRESH_K) -> detector.DetectionResult:
    """Tool: run the default anomaly localization."""
    return detector.detect(prep.gray, scale=prep.scale, thresh_k=thresh_k)


def run_targeted_reinspection(prep: Prepared, thresh_k: float) -> detector.DetectionResult:
    """Tool: re-run detection with an adjusted (lower) threshold to probe
    borderline cases. Only the allowed threshold parameter is changed."""
    return detector.detect(prep.gray, scale=prep.scale, thresh_k=thresh_k)


def compare_analysis_results(a: detector.DetectionResult, b: detector.DetectionResult) -> Dict:
    """Tool: compare two detection runs to see whether evidence strengthened."""
    return {
        "score_delta": round(b.anomaly_score - a.anomaly_score, 4),
        "region_delta": len(b.regions) - len(a.regions),
        "a_regions": len(a.regions),
        "b_regions": len(b.regions),
    }


def request_human_review(reason: str) -> Dict:
    """Tool: escalate to a human with a recorded reason."""
    return {"status": "human_review", "reason": reason}
