"""FastAPI app: health check + image inspection endpoint.

This is the component intended to run on AWS (App Runner / ECS). It validates
input, enforces a size limit, runs the OpenCV + agent pipeline, and returns
structured results plus base64 annotated images. Errors are mapped to clear
HTTP status codes; nothing crashes the worker.
"""
from __future__ import annotations

import os

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from src import __version__
from src.pipeline import bundle_to_json, inspect_bytes
from src.vision.preprocessing import ImageDecodeError, UnsupportedFormatError

import cv2

MAX_UPLOAD_BYTES = int(os.getenv("INSPECTAI_MAX_UPLOAD_BYTES", str(8 * 1024 * 1024)))
MAX_AGENT_ITERS = int(os.getenv("INSPECTAI_MAX_AGENT_ITERS", "4"))

app = FastAPI(
    title="InspectAI",
    description="Agentic Visual Quality Inspector (OpenCV 5 + agentic decision loop).",
    version=__version__,
)


@app.get("/health")
def health() -> dict:
    """Liveness/readiness probe used by AWS health checks."""
    return {
        "status": "ok",
        "service": "inspectai",
        "version": __version__,
        "opencv": cv2.__version__,
    }


@app.post("/inspect")
async def inspect(file: UploadFile = File(...)) -> JSONResponse:
    data = await file.read()

    if not data:
        raise HTTPException(status_code=400, detail="empty upload")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"file too large ({len(data)} > {MAX_UPLOAD_BYTES} bytes)",
        )

    try:
        bundle = inspect_bytes(data, max_iters=MAX_AGENT_ITERS)
    except UnsupportedFormatError as e:
        raise HTTPException(status_code=415, detail=f"unsupported format: {e}")
    except ImageDecodeError as e:
        raise HTTPException(status_code=422, detail=f"could not decode image: {e}")
    except Exception as e:  # defensive: never leak a stack trace / crash worker
        raise HTTPException(status_code=500, detail=f"inspection failed: {e}")

    return JSONResponse(content=bundle_to_json(bundle))
