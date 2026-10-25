"""Reproducible synthetic surface-defect dataset.

We generate textured "product surfaces" (normal) and inject localized defects
(scratches, spots/contamination, dents) with known ground-truth masks. This is
a deliberate, documented stand-in for a gated dataset like MVTec AD, chosen so
evaluation is fully reproducible offline. Limitations are documented in
docs/evaluation.md: synthetic defects are easier than many real-world cases, so
reported metrics are an upper-bound sanity check, not a real-world benchmark.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class Sample:
    image: np.ndarray            # BGR uint8
    mask: np.ndarray             # uint8 {0,255} ground-truth defect mask
    label: int                   # 1 = defective, 0 = normal
    kind: str                    # 'normal' | 'scratch' | 'spot' | 'dent'


def _base_surface(rng: np.random.Generator, size: int = 256) -> np.ndarray:
    """A controlled-lighting product surface: gentle shading + fine grain.

    We assume a controlled inspection setup (even lighting, fixed standoff),
    which is standard for automated visual inspection and keeps the clean
    surface low-contrast relative to genuine defects. This assumption is
    documented in docs/evaluation.md.
    """
    base = int(rng.integers(120, 150))
    # Gentle low-frequency shading (controlled lighting => small amplitude).
    field = rng.normal(0, 8, (size // 16, size // 16)).astype(np.float32)
    field = cv2.resize(field, (size, size), interpolation=cv2.INTER_CUBIC)
    surface = base + field
    # Fine manufacturing grain.
    surface += rng.normal(0, 2.0, (size, size)).astype(np.float32)
    surface = np.clip(surface, 0, 255).astype(np.uint8)
    return cv2.cvtColor(surface, cv2.COLOR_GRAY2BGR)


def _inject_scratch(img, mask, rng):
    h, w = img.shape[:2]
    p1 = (int(rng.integers(10, w - 10)), int(rng.integers(10, h - 10)))
    angle = rng.uniform(0, np.pi)
    length = int(rng.integers(40, 110))
    p2 = (int(p1[0] + length * np.cos(angle)), int(p1[1] + length * np.sin(angle)))
    thickness = int(rng.integers(1, 4))
    color = int(rng.integers(10, 50))  # clearly dark scratch
    cv2.line(img, p1, p2, (color, color, color), thickness, cv2.LINE_AA)
    cv2.line(mask, p1, p2, 255, thickness + 2, cv2.LINE_AA)


def _inject_spot(img, mask, rng):
    h, w = img.shape[:2]
    c = (int(rng.integers(20, w - 20)), int(rng.integers(20, h - 20)))
    r = int(rng.integers(6, 16))
    # Contamination spot: clearly darker or brighter than the ~135 base surface.
    color = int(rng.integers(15, 60)) if rng.random() < 0.5 else int(rng.integers(210, 255))
    cv2.circle(img, c, r, (color, color, color), -1, cv2.LINE_AA)
    cv2.circle(mask, c, r + 1, 255, -1, cv2.LINE_AA)


def _inject_dent(img, mask, rng):
    h, w = img.shape[:2]
    c = (int(rng.integers(25, w - 25)), int(rng.integers(25, h - 25)))
    ax = (int(rng.integers(10, 22)), int(rng.integers(6, 16)))
    ang = int(rng.integers(0, 180))
    overlay = img.copy()
    cv2.ellipse(overlay, c, ax, ang, 0, 360, (55, 55, 55), -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)  # pronounced dent shadow
    cv2.ellipse(mask, c, ax, ang, 0, 360, 255, -1, cv2.LINE_AA)


_INJECTORS = {"scratch": _inject_scratch, "spot": _inject_spot, "dent": _inject_dent}


def make_sample(seed: int, defect: Optional[str] = None, size: int = 256) -> Sample:
    """Create one labeled sample. defect=None => normal; else a defect kind."""
    rng = np.random.default_rng(seed)
    img = _base_surface(rng, size)
    mask = np.zeros((size, size), dtype=np.uint8)
    if defect is None:
        return Sample(image=img, mask=mask, label=0, kind="normal")
    _INJECTORS[defect](img, mask, rng)
    return Sample(image=img, mask=mask, label=1, kind=defect)


def make_dataset(n_normal: int = 30, n_defect: int = 30, seed: int = 7) -> List[Sample]:
    """Deterministic balanced dataset. Defect kinds cycle for variety."""
    kinds = ["scratch", "spot", "dent"]
    samples: List[Sample] = []
    for i in range(n_normal):
        samples.append(make_sample(seed * 1000 + i, defect=None))
    for j in range(n_defect):
        kind = kinds[j % len(kinds)]
        samples.append(make_sample(seed * 2000 + j, defect=kind))
    return samples


def encode_png(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("failed to PNG-encode sample")
    return buf.tobytes()
