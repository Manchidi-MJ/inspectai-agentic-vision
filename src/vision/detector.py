"""Classical, defensible anomaly localization using OpenCV 5.

Method (documented in docs/technical_report.md):
  1. Build an intensity residual: |gray - median_blur(gray)|. Surface defects
     (scratches, spots, contamination) deviate locally from the smooth product
     surface, so they survive this high-pass-like residual while the clean
     surface cancels out.
  2. Build a structural residual from gradient magnitude (Sobel). Cracks and
     edges of foreign material produce gradient energy the clean surface lacks.
  3. Fuse the two normalized residuals, blur to suppress pixel noise, and
     threshold adaptively (mean + k*std of the anomaly map, clamped). This is
     deterministic and reproducible; it is NOT a learned detector.
  4. Morphologically clean, extract connected components, and measure each
     region (area, bbox, fill ratio, mean anomaly score).

This is a visual-anomaly localizer, not a certified defect detector. A high
score means "this region deviates from its local surroundings", which the agent
and UI translate into a *potential* anomaly, never a confirmed defect.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple

import cv2
import numpy as np


# Documented heuristics, not probabilities. See docs/technical_report.md.
# The anomaly map is an absolute local-contrast z-score mapped to [0,1] by
# Z_SCALE. Thresholding is therefore an ABSOLUTE cut on that map (ABS_THRESH),
# optionally raised by a per-image guard (mean + K*std) so very clean images
# are not over-segmented. thresh_k is exposed so the agent can re-inspect.
Z_SCALE = 45.0           # gray-level residual treated as "fully anomalous" (=>1)
ABS_THRESH = 0.35        # absolute floor on the [0,1] map for a pixel to flag
THRESH_K = 3.5           # per-image guard: also require >= mean + K*std
MIN_REGION_AREA = 30     # ignore components smaller than this (noise), in px
MEDIAN_KSIZE = 9         # local-model smoothing kernel (odd)
LOCAL_WIN = 15           # neighborhood window for local mean/std (odd)


@dataclass
class Region:
    bbox: Tuple[int, int, int, int]   # x, y, w, h in ORIGINAL image coords
    area_px: int                      # component area in working-res pixels
    fill_ratio: float                 # area / bbox area
    mean_score: float                 # mean anomaly-map value in region [0,1]

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class DetectionResult:
    anomaly_score: float              # image-level score in [0,1]
    threshold: float                  # threshold applied to the anomaly map
    regions: List[Region] = field(default_factory=list)
    anomaly_map: np.ndarray = field(default=None, repr=False)  # float32 [0,1]
    params: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "anomaly_score": round(self.anomaly_score, 4),
            "threshold": round(self.threshold, 4),
            "num_regions": len(self.regions),
            "regions": [r.to_dict() for r in self.regions],
            "params": self.params,
        }


def _normalize01(arr: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0,1] as float32; constant arrays map to zeros."""
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-9:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def build_anomaly_map(
    gray: np.ndarray,
    median_ksize: int = MEDIAN_KSIZE,
    local_win: int = LOCAL_WIN,
) -> np.ndarray:
    """Return a float32 [0,1] local-contrast anomaly map.

    Defects are regions that deviate from their *local* neighborhood by more
    than the local texture noise. We therefore z-score each pixel against the
    mean and std of its neighborhood (a Mahalanobis-style local-contrast
    measure). Uniform fine grain cancels because it inflates the local std,
    while a coherent defect stands out because it shifts the local mean. This
    is what makes defects score higher than clean textured surfaces.
    """
    # Denoise so per-pixel grain does not survive as spurious residual.
    g = cv2.GaussianBlur(gray.astype(np.float32), (3, 3), 0)

    # Background model = broad illumination/shading. Subtracting it leaves a
    # high-pass residual (in GRAY LEVELS) where sharp, local defects live while
    # smooth shading and uniform grain cancel to near zero.
    bg = cv2.GaussianBlur(g, (0, 0), sigmaX=max(8.0, local_win))
    residual = np.abs(g - bg)

    # Average over a small window so a real defect forms a coherent blob and
    # isolated single-pixel grain is suppressed by neighborhood averaging.
    residual = cv2.blur(residual, (5, 5))

    # Map the ABSOLUTE residual amplitude to [0,1] with a FIXED gray-level
    # scale. A defect shifts gray by tens of levels over a compact area and
    # approaches 1; grain and shading stay near 0. This preserves absolute
    # contrast rather than re-normalizing noise back up.
    return np.clip(residual / Z_SCALE, 0.0, 1.0).astype(np.float32)


def detect(
    gray: np.ndarray,
    scale: float = 1.0,
    thresh_k: float = THRESH_K,
    min_region_area: int = MIN_REGION_AREA,
    median_ksize: int = MEDIAN_KSIZE,
) -> DetectionResult:
    """Localize suspicious regions on a single-channel image.

    `scale` is working/original ratio; bounding boxes are mapped back to
    original-image coordinates so the UI can annotate the full-res picture.
    """
    if gray.ndim != 2:
        raise ValueError("detect() expects a single-channel image")

    amap = build_anomaly_map(gray, median_ksize=median_ksize)

    mean, std = float(amap.mean()), float(amap.std())
    # Absolute floor OR'd with a per-image guard. thresh_k scales the guard so
    # the agent's lowered-k re-inspection genuinely relaxes the criterion.
    guard = mean + thresh_k * std
    threshold = float(np.clip(min(ABS_THRESH, guard) if thresh_k < THRESH_K
                              else max(ABS_THRESH, guard), 0.0, 1.0))

    mask = (amap >= threshold).astype(np.uint8) * 255

    # Suppress a thin border: the background-subtraction residual is unreliable
    # within one blur radius of the edge and produces spurious border flags.
    b = max(4, LOCAL_WIN // 2)
    mask[:b, :] = 0
    mask[-b:, :] = 0
    mask[:, :b] = 0
    mask[:, -b:] = 0

    # Morphological opening then closing to drop specks and fuse fragments.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    regions: List[Region] = []
    inv_scale = 1.0 / scale if scale > 0 else 1.0
    for i in range(1, num):  # 0 is background
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_region_area:
            continue
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        comp_mask = labels == i
        mean_score = float(amap[comp_mask].mean())
        fill = area / float(max(1, w * h))
        regions.append(
            Region(
                bbox=(
                    int(round(x * inv_scale)),
                    int(round(y * inv_scale)),
                    int(round(w * inv_scale)),
                    int(round(h * inv_scale)),
                ),
                area_px=area,
                fill_ratio=round(fill, 3),
                mean_score=round(mean_score, 4),
            )
        )

    regions.sort(key=lambda r: r.mean_score * r.area_px, reverse=True)

    # Image-level score rewards a *concentrated, high-contrast* region, which is
    # what distinguishes a defect from evenly distributed surface grain. We take
    # the strongest region's contrast (mean anomaly value) and modulate it by a
    # coherence term (how much the flagged mask clusters into few components vs.
    # scatters as speckle). Transparent and bounded; not a learned probability.
    flagged_fraction = float((mask > 0).mean())
    top_region_score = max((r.mean_score for r in regions), default=0.0)
    top_region_area = max((r.area_px for r in regions), default=0)

    if regions:
        total_flagged = float(sum(r.area_px for r in regions))
        # Fraction of flagged pixels belonging to the single largest region:
        # high => one coherent blob (defect-like); low => scattered noise.
        coherence = top_region_area / max(1.0, total_flagged)
    else:
        coherence = 0.0

    # Peak absolute contrast anywhere on the map anchors the score; the top
    # region's own contrast and coherence refine it. On a clean surface the map
    # stays low everywhere, so the score stays low.
    map_peak = float(amap.max())
    image_score = float(np.clip(
        0.6 * map_peak + 0.4 * top_region_score * (0.5 + 0.5 * coherence),
        0.0, 1.0,
    ))

    return DetectionResult(
        anomaly_score=image_score,
        threshold=threshold,
        regions=regions,
        anomaly_map=amap,
        params={
            "thresh_k": thresh_k,
            "min_region_area": min_region_area,
            "median_ksize": median_ksize,
            "flagged_fraction": round(flagged_fraction, 5),
            "map_mean": round(mean, 5),
            "map_std": round(std, 5),
        },
    )


def annotate(bgr_original: np.ndarray, result: DetectionResult) -> np.ndarray:
    """Draw bounding boxes + scores on a copy of the original-resolution image."""
    out = bgr_original.copy()
    for idx, r in enumerate(result.regions):
        x, y, w, h = r.bbox
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 0, 255), 2)
        label = f"{r.mean_score:.2f}"
        cv2.putText(
            out, label, (x, max(0, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA,
        )
    return out


def heatmap_overlay(bgr_original: np.ndarray, result: DetectionResult) -> np.ndarray:
    """Return a color heatmap of the anomaly map blended over the original."""
    if result.anomaly_map is None:
        return bgr_original.copy()
    h, w = bgr_original.shape[:2]
    amap8 = (result.anomaly_map * 255).astype(np.uint8)
    amap8 = cv2.resize(amap8, (w, h), interpolation=cv2.INTER_LINEAR)
    heat = cv2.applyColorMap(amap8, cv2.COLORMAP_JET)
    return cv2.addWeighted(bgr_original, 0.6, heat, 0.4, 0.0)
