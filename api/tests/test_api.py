"""M4.2 Phase K: API endpoint tests. Uses FastAPI's TestClient (no
actual server process needed). Run with KMP_DUPLICATE_LIB_OK=TRUE (the
API imports both torch and engine.image_io -- see api/main.py's module
docstring for why)."""

from __future__ import annotations

import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from api.main import app  # noqa: E402

client = TestClient(app)

SYNTH_IMAGE = os.path.join(os.path.dirname(__file__), "..", "..", "synthetic_photos", "kolam19_k1.jpg")
pytestmark = pytest.mark.skipif(not os.path.exists(SYNTH_IMAGE), reason="synthetic_photos/ fixture missing")


def test_health():
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["classical_detector_available"] is True
    assert "gated_detector_available" in body


def test_model_info():
    r = client.get("/api/v1/model")
    assert r.status_code == 200
    body = r.json()
    assert "classical_detector" in body
    assert "ml_checkpoint_exists" in body
    assert "gated_model_version" in body


def test_detect_classical_default():
    with open(SYNTH_IMAGE, "rb") as f:
        r = client.post("/api/v1/detect", files={"image": ("k.jpg", f, "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is True
    assert body["detector"] == "classical"
    assert body["count"] == len(body["detections"])
    assert body["image"]["width"] > 0 and body["image"]["height"] > 0
    # coordinates must be in ORIGINAL image space, not model-input/heatmap space
    for det in body["detections"]:
        assert 0 <= det["x"] <= body["image"]["width"] * 1.05  # small tolerance for deskew edge effects
        assert 0 <= det["y"] <= body["image"]["height"] * 1.05


def test_detect_explicit_detector_selection():
    with open(SYNTH_IMAGE, "rb") as f:
        r = client.post("/api/v1/detect", files={"image": ("k.jpg", f, "image/jpeg")}, data={"detector": "classical"})
    assert r.status_code == 200
    assert r.json()["detector"] == "classical"


def test_detect_ml_gated():
    """Verify /detect accepts detector=ml-gated and propagates it correctly."""
    with open(SYNTH_IMAGE, "rb") as f:
        r = client.post("/api/v1/detect", files={"image": ("k.jpg", f, "image/jpeg")}, data={"detector": "ml-gated"})
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is True
    assert body["detector"] == "ml-gated"


def test_detect_invalid_detector_name():
    with open(SYNTH_IMAGE, "rb") as f:
        r = client.post("/api/v1/detect", files={"image": ("k.jpg", f, "image/jpeg")}, data={"detector": "quantum"})
    assert r.status_code == 400
    assert r.json()["success"] is False


def test_detect_malformed_upload_rejected():
    r = client.post("/api/v1/detect", files={"image": ("bad.txt", b"not an image", "text/plain")})
    assert r.status_code == 400
    assert r.json()["success"] is False


def test_detect_empty_upload_rejected():
    r = client.post("/api/v1/detect", files={"image": ("empty.jpg", b"", "image/jpeg")})
    assert r.status_code == 400


def test_analyze_returns_graph_motifs_validity():
    with open(SYNTH_IMAGE, "rb") as f:
        r = client.post("/api/v1/analyze", files={"image": ("k.jpg", f, "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is True
    assert "graph" in body and set(body["graph"].keys()) == {"nodes", "edges", "distinct_edges"}
    assert "motifs" in body and "motif_count" in body["motifs"]
    assert "validity" in body and "is_eulerian_circuit" in body["validity"]
    # no networkx object leaked -- every value must be JSON primitive
    import json
    json.dumps(body)  # raises if anything non-serializable slipped through


def test_analyze_ml_gated():
    """Verify /analyze supports detector=ml-gated through the full pipeline."""
    with open(SYNTH_IMAGE, "rb") as f:
        r = client.post("/api/v1/analyze", files={"image": ("k.jpg", f, "image/jpeg")}, data={"detector": "ml-gated"})
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is True
    assert body["detector"] == "ml-gated"


def test_reconstruct_returns_structured_result():
    with open(SYNTH_IMAGE, "rb") as f:
        r = client.post("/api/v1/reconstruct", files={"image": ("k.jpg", f, "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is True
    assert "reconstruction" in body
    import json
    json.dumps(body)


def test_reconstruct_ml_gated():
    """Verify /reconstruct supports detector=ml-gated."""
    with open(SYNTH_IMAGE, "rb") as f:
        r = client.post("/api/v1/reconstruct", files={"image": ("k.jpg", f, "image/jpeg")}, data={"detector": "ml-gated"})
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is True
    assert body["detector"] == "ml-gated"


def test_compare_detectors_reports_both_sides():
    with open(SYNTH_IMAGE, "rb") as f:
        r = client.post("/api/v1/compare-detectors", files={"image": ("k.jpg", f, "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is True
    assert body["classical"]["detector"] == "classical"
    assert body["ml"]["detector"] == "ml"
    assert "agreement" in body


def test_detect_default_is_classical_not_ml():
    """Production-default rule: omitting `detector` must select
    classical, never ml, per the task's explicit fallback-behavior rule."""
    with open(SYNTH_IMAGE, "rb") as f:
        r = client.post("/api/v1/detect", files={"image": ("k.jpg", f, "image/jpeg")})
    assert r.json()["detector"] == "classical"


def test_no_silent_fallback_on_ml_gated_failure(monkeypatch):
    """CRITICAL RULE: When detector=ml-gated fails (missing checkpoint/inference error),
    the API MUST return HTTP 503 and NEVER silently fall back to classical."""
    from api.detectors import _ML_GATED, _CLASSICAL

    classical_called = False
    original_detect = _CLASSICAL.detect

    def mock_classical_detect(*args, **kwargs):
        nonlocal classical_called
        classical_called = True
        return original_detect(*args, **kwargs)

    def mock_gated_detect(*args, **kwargs):
        raise RuntimeError("ML model checkpoint missing")

    monkeypatch.setattr(_CLASSICAL, "detect", mock_classical_detect)
    monkeypatch.setattr(_ML_GATED, "detect", mock_gated_detect)

    with open(SYNTH_IMAGE, "rb") as f:
        r = client.post("/api/v1/detect", files={"image": ("k.jpg", f, "image/jpeg")}, data={"detector": "ml-gated"})

    assert r.status_code == 503
    assert r.json()["success"] is False
    assert "ML model checkpoint missing" in r.json()["error"]
    assert classical_called is False, "CRITICAL ERROR: API silently fell back to classical!"

