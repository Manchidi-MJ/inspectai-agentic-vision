"""Image-quality assessment with transparent, validated measures.

Every metric here is a well-known image statistic, not a learned "confidence".
We report the raw measurement and a boolean flag against a documented
threshold, and we never convert these into a probability of a defect.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List

import cv2
import numpy as np


# Documented thresholds. These are heuristics tuned on the synthetic dataset
# (see docs/evaluation.md), NOT calibrated probabilities. They flag images
# that are likely too degraded for a reliable inspection.
# Calibrated to a controlled inspection surface (even lighting, fixed standoff);
# see docs/evaluation.md. These flag images too degraded to inspect reliably.
BLUR_VAR_MIN = 25.0        # variance of Laplacian below this => likely blurry
BRIGHT_LOW = 35.0          # mean luma below => underexposed
BRIGHT_HIGH = 225.0        # mean luma above => overexposed
CONTRAST_STD_MIN = 4.0     # std of luma below => flat/featureless frame
SATURATION_FRAC_MAX = 0.25 # fraction of clipped pixels above => over-saturated


@dataclass
class QualityReport:
    blur_laplacian_var: float
    brightness_mean: float
    contrast_std: float
    clipped_fraction: float
    flags: List[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        """True when no quality flag fired."""
        return len(self.flags) == 0

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["usable"] = self.usable
        return d


def assess(gray: np.ndarray) -> QualityReport:
    """Compute quality metrics on a single-channel uint8 image."""
    if gray.ndim != 2:
        raise ValueError("assess() expects a single-channel image")

    # Focus / sharpness: variance of the Laplacian (standard no-reference measure).
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    blur_var = float(lap.var())

    brightness = float(gray.mean())
    contrast = float(gray.std())

    # Clipping: pixels stuck at 0 or 255 indicate exposure problems.
    clipped = float(np.count_nonzero((gray == 0) | (gray == 255))) / gray.size

    flags: List[str] = []
    if blur_var < BLUR_VAR_MIN:
        flags.append("blurry")
    if brightness < BRIGHT_LOW:
        flags.append("underexposed")
    if brightness > BRIGHT_HIGH:
        flags.append("overexposed")
    if contrast < CONTRAST_STD_MIN:
        flags.append("low_contrast")
    if clipped > SATURATION_FRAC_MAX:
        flags.append("clipped")

    return QualityReport(
        blur_laplacian_var=round(blur_var, 3),
        brightness_mean=round(brightness, 3),
        contrast_std=round(contrast, 3),
        clipped_fraction=round(clipped, 5),
        flags=flags,
    )
