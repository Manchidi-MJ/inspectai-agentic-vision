from fastapi.testclient import TestClient

from src.api.routes import app
from src.vision import synthetic

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["opencv"].startswith("5.")


def test_inspect_valid_defect_image():
    s = synthetic.make_sample(31, defect="spot")
    png = synthetic.encode_png(s.image)
    r = client.post("/inspect", files={"file": ("x.png", png, "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in (
        "potential_anomaly", "no_anomaly_detected", "human_review_required"
    )
    assert "images" in body and "annotated_png_b64" in body["images"]
    assert len(body["decision_log"]) >= 2


def test_inspect_empty_upload():
    r = client.post("/inspect", files={"file": ("x.png", b"", "image/png")})
    assert r.status_code == 400


def test_inspect_unsupported_format():
    gif = b"GIF89a" + b"\x00" * 16
    r = client.post("/inspect", files={"file": ("x.gif", gif, "image/gif")})
    assert r.status_code == 415


def test_inspect_corrupt_image():
    corrupt = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
    r = client.post("/inspect", files={"file": ("x.png", corrupt, "image/png")})
    assert r.status_code == 422


def test_inspect_oversize(monkeypatch):
    import src.api.routes as routes
    monkeypatch.setattr(routes, "MAX_UPLOAD_BYTES", 10)
    s = synthetic.make_sample(32, defect="spot")
    png = synthetic.encode_png(s.image)
    r = client.post("/inspect", files={"file": ("x.png", png, "image/png")})
    assert r.status_code == 413
