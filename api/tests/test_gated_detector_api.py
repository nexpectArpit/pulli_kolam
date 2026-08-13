"""Unit tests for GatedMLDetector and get_detector factory normalization.
Tests detector behavior, DetectionResult, graph structure, original image coordinate un-deskewing,
and explicit error handling."""

from __future__ import annotations

import os
import sys
import pytest
import numpy as np
import networkx as nx

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from api.detectors import get_detector, GatedMLDetector, DetectionResult, _undo_deskew  # noqa: E402

SYNTH_IMAGE = os.path.join(os.path.dirname(__file__), "..", "..", "synthetic_photos", "kolam19_k1.jpg")


def test_factory_normalization():
    """Verify that get_detector handles aliases ('ml-gated', 'ml_gated', 'gated_ml', 'gated-ml')
    and normalizes them to the same GatedMLDetector instance."""
    d1 = get_detector("ml-gated")
    d2 = get_detector("ml_gated")
    d3 = get_detector("gated_ml")
    d4 = get_detector("gated-ml")
    d5 = get_detector("  ML-GATED  ")

    assert isinstance(d1, GatedMLDetector)
    assert d1 is d2
    assert d2 is d3
    assert d3 is d4
    assert d4 is d5


def test_factory_unsupported_detector_rejected():
    """Verify that unknown detector strings raise a clear ValueError."""
    with pytest.raises(ValueError, match="unknown detector"):
        get_detector("invalid-detector")

    with pytest.raises(ValueError, match="unknown detector"):
        get_detector(123)  # type: ignore


def test_undo_deskew_identity_when_zero_rotation():
    """Verify _undo_deskew returns identical coordinates when rotation_deg == 0."""
    pts = [(10.0, 20.0), (30.0, 40.0)]
    undone = _undo_deskew(pts, 0.0, 100, 100)
    assert undone == pts


def test_undo_deskew_transforms_points_correctly():
    """Verify _undo_deskew transforms coordinates back when rotation is applied."""
    pts = [(50.0, 60.0)]
    undone = _undo_deskew(pts, 90.0, 100, 100)
    assert len(undone) == 1
    # Rotate (50, 60) by -90 deg around center (50, 50) -> (40.0, 50.0) in image space
    assert pytest.approx(undone[0][0], abs=1.0) == 40.0
    assert pytest.approx(undone[0][1], abs=1.0) == 50.0



@pytest.mark.skipif(not os.path.exists(SYNTH_IMAGE), reason="synthetic_photos fixture missing")
def test_gated_detector_execution_on_synthetic_image():
    """Verify GatedMLDetector processes a valid image, producing DetectionResult with graph and original coords."""
    detector = get_detector("ml-gated")
    result = detector.detect(SYNTH_IMAGE)

    assert isinstance(result, DetectionResult)
    assert result.detector == "ml-gated"
    assert result.model_version is not None
    assert result.width > 0 and result.height > 0
    assert result.processing_ms > 0
    assert isinstance(result.graph, nx.MultiGraph)
    assert len(result.dots) == len(result.graph.nodes())

    for x, y in result.dots:
        assert 0 <= x <= result.width * 1.05
        assert 0 <= y <= result.height * 1.05


def test_gated_detector_missing_file_raises():
    """Verify GatedMLDetector raises FileNotFoundError when image file is missing."""
    detector = GatedMLDetector()
    with pytest.raises(FileNotFoundError, match="could not read image"):
        detector.detect("non_existent_file_path_12345.jpg")
