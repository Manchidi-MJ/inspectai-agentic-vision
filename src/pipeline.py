"""Single source of truth for the end-to-end inspection pipeline.

Both the API and the Streamlit UI call inspect_bytes() so behavior is identical
whether run locally, in the UI, or on AWS.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Dict, Optional

import cv2
import numpy as np

from src.agent.controller import InspectionAgent
from src.vision import detector, preprocessing


@dataclass
class InspectionBundle:
    result: Dict                      # agent outcome (JSON-serializable)
    annotated_png: bytes              # original with region boxes
    heatmap_png: bytes                # anomaly heatmap overlay
    original_png: bytes               # re-encoded original (normalized decode)


def _png(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("PNG encode failed")
    return buf.tobytes()


def inspect_bytes(data: bytes, max_iters: int = 4) -> InspectionBundle:
    """Decode -> normalize -> agent inspect -> render annotations."""
    img = preprocessing.decode_image(data)          # may raise (handled by caller)
    prep = preprocessing.normalize(img)

    agent = InspectionAgent(max_iters=max_iters)
    outcome = agent.run(prep)

    # Re-run detection once for rendering (deterministic; same params as agent's
    # primary analysis) so we can draw boxes/heatmap on the original image.
    det = detector.detect(prep.gray, scale=prep.scale)
    annotated = detector.annotate(img, det)
    heat = detector.heatmap_overlay(img, det)

    return InspectionBundle(
        result=outcome.to_dict(),
        annotated_png=_png(annotated),
        heatmap_png=_png(heat),
        original_png=_png(img),
    )


def bundle_to_json(bundle: InspectionBundle) -> Dict:
    """Serialize a bundle for an HTTP response (images base64-encoded)."""
    return {
        **bundle.result,
        "images": {
            "original_png_b64": base64.b64encode(bundle.original_png).decode(),
            "annotated_png_b64": base64.b64encode(bundle.annotated_png).decode(),
            "heatmap_png_b64": base64.b64encode(bundle.heatmap_png).decode(),
        },
    }
