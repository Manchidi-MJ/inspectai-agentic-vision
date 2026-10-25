"""InspectAI — Streamlit front end.

Runs the inspection pipeline either in-process (default) or against a remote
AWS-hosted API if INSPECTAI_API_URL is set. The UI shows the original and
annotated images, suspicious regions, measurements, image-quality warnings,
the full agent decision trace, and the final status — with explicit warnings
that automated findings require human verification.
"""
from __future__ import annotations

import base64
import io
import os
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import inspect_bytes  # noqa: E402
from src.vision.preprocessing import ImageDecodeError, UnsupportedFormatError  # noqa: E402
from src.vision import synthetic  # noqa: E402

API_URL = os.getenv("INSPECTAI_API_URL", "").strip()
SAMPLES_DIR = Path(__file__).parent / "samples"

STATUS_STYLE = {
    "potential_anomaly": ("⚠️ Potential anomaly", "#b45309"),
    "no_anomaly_detected": ("✅ No anomaly detected by this system", "#15803d"),
    "human_review_required": ("🧑‍🔬 Human review required", "#b91c1c"),
}

st.set_page_config(page_title="InspectAI", page_icon="🔎", layout="wide")


def _decode_b64_png(b64: str) -> np.ndarray:
    buf = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def _run_remote(data: bytes) -> dict:
    import httpx
    r = httpx.post(
        f"{API_URL}/inspect",
        files={"file": ("upload.png", data, "image/png")},
        timeout=60.0,
    )
    r.raise_for_status()
    return r.json()


def _run_local(data: bytes) -> dict:
    from src.pipeline import bundle_to_json
    return bundle_to_json(inspect_bytes(data))


def run_inspection(data: bytes) -> dict:
    return _run_remote(data) if API_URL else _run_local(data)


# ---------------------------------------------------------------- sidebar ----
st.sidebar.title("🔎 InspectAI")
st.sidebar.caption("Agentic Visual Quality Inspector")
st.sidebar.markdown(
    f"**Engine:** OpenCV {cv2.__version__}  \n"
    f"**Mode:** {'Remote API' if API_URL else 'In-process pipeline'}"
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**How to use**\n\n"
    "1. Upload an image or pick a sample.\n"
    "2. Click **Inspect image**.\n"
    "3. Review regions, the agent trace, and the final status.\n"
)
st.sidebar.info(
    "Automated findings are **decision support**, not a certified pass/fail. "
    "A flagged region is a *potential* visual anomaly and requires human "
    "verification."
)

st.title("InspectAI — See the defect. Investigate the evidence. Decide what happens next.")

# ------------------------------------------------------------- input area ----
col_in, col_prev = st.columns([1, 1])

data: bytes | None = None
with col_in:
    st.subheader("Input")
    uploaded = st.file_uploader("Upload an image (PNG/JPG/BMP)", type=["png", "jpg", "jpeg", "bmp"])
    st.markdown("**or generate a reproducible sample:**")
    sample_kind = st.selectbox("Sample kind", ["(none)", "normal", "scratch", "spot", "dent"])
    sample_seed = st.number_input("Sample seed", min_value=0, value=101, step=1)
    if uploaded is not None:
        data = uploaded.read()
    elif sample_kind != "(none)":
        s = synthetic.make_sample(int(sample_seed), defect=None if sample_kind == "normal" else sample_kind)
        data = synthetic.encode_png(s.image)

with col_prev:
    st.subheader("Preview")
    if data:
        arr = np.frombuffer(data, np.uint8)
        prev = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if prev is not None:
            h, w = prev.shape[:2]
            st.image(cv2.cvtColor(prev, cv2.COLOR_BGR2RGB), caption=f"{w}×{h}px, {len(data)/1024:.1f} KB")
    else:
        st.caption("No image selected yet.")

go = st.button("🔬 Inspect image", type="primary", disabled=data is None)

# ----------------------------------------------------------------- results ----
if go and data:
    try:
        result = run_inspection(data)
    except UnsupportedFormatError as e:
        st.error(f"Unsupported format: {e}")
        st.stop()
    except ImageDecodeError as e:
        st.error(f"Could not decode image: {e}")
        st.stop()
    except Exception as e:  # pragma: no cover - UI guard
        st.error(f"Inspection failed: {e}")
        st.stop()

    status = result["status"]
    label, color = STATUS_STYLE.get(status, (status, "#334155"))
    st.markdown(
        f"<div style='padding:14px;border-radius:10px;background:{color};"
        f"color:white;font-size:20px;font-weight:600'>{label}</div>",
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4 = st.columns(4)
    score = result.get("anomaly_score")
    m1.metric("Anomaly score", f"{score:.3f}" if score is not None else "n/a")
    m2.metric("Suspicious regions", result.get("num_regions", 0))
    m3.metric("Agent iterations", result.get("iterations", 0))
    m4.metric("Escalated", "Yes" if result.get("escalated") else "No")

    # Images.
    imgs = result.get("images", {})
    if imgs:
        c1, c2, c3 = st.columns(3)
        c1.image(_decode_b64_png(imgs["original_png_b64"]), caption="Original")
        c2.image(_decode_b64_png(imgs["annotated_png_b64"]), caption="Annotated regions")
        c3.image(_decode_b64_png(imgs["heatmap_png_b64"]), caption="Anomaly heatmap")

    # Quality warnings.
    q = result.get("quality", {})
    if q.get("flags"):
        st.warning("Image-quality flags: " + ", ".join(q["flags"]))
    with st.expander("Image-quality measurements"):
        st.json(q)

    # Regions table.
    regions = result.get("regions", [])
    if regions:
        st.subheader("Suspicious region measurements")
        st.dataframe(
            [
                {
                    "x": r["bbox"][0], "y": r["bbox"][1],
                    "w": r["bbox"][2], "h": r["bbox"][3],
                    "area_px": r["area_px"], "fill_ratio": r["fill_ratio"],
                    "mean_score": r["mean_score"],
                }
                for r in regions
            ],
            use_container_width=True,
        )

    # Decision trace.
    st.subheader("Agent decision trace")
    for step in result.get("decision_log", []):
        with st.expander(
            f"#{step['iteration']} · {step['phase']}"
            + (f" · action={step['action']}" if step.get("action") else "")
        ):
            if step.get("rationale"):
                st.markdown(f"**Why:** {step['rationale']}")
            if step.get("action_params"):
                st.markdown(f"**Params:** `{step['action_params']}`")
            st.json({"observation": step.get("observation"), "result": step.get("result")})

    st.caption(
        "InspectAI localizes *visual anomalies* using classical OpenCV analysis. "
        "A flagged region is not a confirmed defect. Confirm all findings with a "
        "qualified human inspector before acting."
    )
