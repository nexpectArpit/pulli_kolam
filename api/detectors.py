"""M4.2 Phase F/G: the detector abstraction the API layer uses so it
never has to know whether dots came from the classical detector or an
ML detector. Lives here (api/), NOT in engine/ -- engine/ stays free of
PyTorch/detector-specific code, matching the boundary
engine/ml_contract.py already establishes (a Protocol only, no concrete
implementation). This is intentionally thin -- one dataclass, two small
classes -- per the task's "do not over-engineer this" instruction.

COORDINATE CONTRACT: `DetectionResult.dots` are in the ORIGINAL,
AS-UPLOADED image's pixel coordinate frame -- not the internal
model-input or heatmap-cell frame, and not even
`engine.image_io.preprocess()`'s own deskewed-and-rotated frame.
`preprocess()` rotates the image around its own center to correct mild
camera tilt; both detectors' raw output lands in that ROTATED frame, so
this module explicitly rotates back (inverse of `rotation_deg`) before
returning -- otherwise dot overlays would not align with the image the
user actually sees. This is the "map coordinates back to ORIGINAL IMAGE
coordinates" requirement, applied precisely, not just for the
model-input resize.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field

import cv2
import networkx as nx
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine import image_io  # noqa: E402


@dataclass
class DetectionResult:
    detector: str  # "classical" | "ml"
    model_version: str | None
    dots: list[tuple[float, float]]  # ORIGINAL image pixel coords (x, y) -- see module docstring
    width: int
    height: int
    processing_ms: float
    graph: nx.MultiGraph  # lattice-coordinate graph, for /analyze and /reconstruct -- NOT serialized directly to JSON
    error: str | None = field(default=None)  # set instead of raising, for /compare-detectors partial-failure reporting


def _undo_deskew(pixel_positions: list[tuple[float, float]], rotation_deg: float, w: int, h: int) -> list[tuple[float, float]]:
    if not pixel_positions or rotation_deg == 0.0:
        return list(pixel_positions)
    M_inv = cv2.getRotationMatrix2D((w / 2, h / 2), -rotation_deg, 1.0)
    pts = np.array(pixel_positions, dtype=np.float32).reshape(-1, 1, 2)
    undone = cv2.transform(pts, M_inv).reshape(-1, 2)
    return [(float(x), float(y)) for x, y in undone]


class ClassicalDetector:
    """Wraps engine.image_io.detect_lattice + trace_path, UNMODIFIED --
    this class adds no new detection logic, only the DetectionResult
    packaging and coordinate un-deskewing described above."""

    name = "classical"
    model_version = None

    def detect(self, image_path: str) -> DetectionResult:
        t0 = time.monotonic()
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"could not read image: {image_path}")
        h, w = img.shape[:2]

        preprocessed = image_io.preprocess(image_path)
        lattice = image_io.detect_lattice(preprocessed)
        edges = image_io.trace_path(preprocessed, lattice)

        graph = nx.MultiGraph()
        graph.add_nodes_from(lattice.lattice_coords)
        for a, b in edges:
            graph.add_edge(a, b)

        dots_original = _undo_deskew(lattice.pixel_positions, preprocessed.rotation_deg, w, h)
        elapsed_ms = (time.monotonic() - t0) * 1000

        return DetectionResult(
            detector=self.name, model_version=self.model_version, dots=dots_original,
            width=w, height=h, processing_ms=elapsed_ms, graph=graph,
        )


class MLDetector:
    """Wraps experiments/m4_2/ml_lattice_detector.py's
    LearnedLatticeDetectorV2. Raises explicitly (does NOT silently fall
    back to classical) if the checkpoint is missing or inference fails
    -- per the task's rules 10/11. Loaded lazily and cached on first use
    so importing this module never requires torch/a checkpoint to
    exist unless the ML detector is actually requested."""

    name = "ml"

    def __init__(self):
        self._detector = None
        self.model_version = None

    def _ensure_loaded(self):
        if self._detector is None:
            from experiments.m4_2.ml_lattice_detector import LearnedLatticeDetectorV2, MalformedOutputError
            try:
                self._detector = LearnedLatticeDetectorV2()
            except MalformedOutputError as e:
                raise RuntimeError(f"ML detector unavailable: {e}") from e
            self.model_version = self._detector.model_version

    def detect(self, image_path: str) -> DetectionResult:
        self._ensure_loaded()
        t0 = time.monotonic()
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"could not read image: {image_path}")
        h, w = img.shape[:2]

        preprocessed = image_io.preprocess(image_path)
        lattice = self._detector(preprocessed)
        edges = image_io.trace_path(preprocessed, lattice)

        graph = nx.MultiGraph()
        graph.add_nodes_from(lattice.lattice_coords)
        for a, b in edges:
            graph.add_edge(a, b)

        dots_original = _undo_deskew(lattice.pixel_positions, preprocessed.rotation_deg, w, h)
        elapsed_ms = (time.monotonic() - t0) * 1000

        return DetectionResult(
            detector=self.name, model_version=self.model_version, dots=dots_original,
            width=w, height=h, processing_ms=elapsed_ms, graph=graph,
        )


class GatedMLDetector:
    """Wraps experiments/m4_2/gated_ml_lattice_detector.py's
    GatedLearnedLatticeDetectorV2. Raises explicitly (does NOT silently fall
    back to classical) if the checkpoint is missing or inference fails.
    Loaded lazily and cached on first use."""

    name = "ml-gated"

    def __init__(self):
        self._detector = None
        self.model_version = None

    def _ensure_loaded(self):
        if self._detector is None:
            from experiments.m4_2.gated_ml_lattice_detector import GatedLearnedLatticeDetectorV2
            from experiments.m4_2.ml_lattice_detector import MalformedOutputError
            try:
                self._detector = GatedLearnedLatticeDetectorV2()
            except (MalformedOutputError, RuntimeError, FileNotFoundError) as e:
                raise RuntimeError(f"Gated ML detector unavailable: {e}") from e
            self.model_version = self._detector.model_version

    def detect(self, image_path: str) -> DetectionResult:
        self._ensure_loaded()
        t0 = time.monotonic()
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"could not read image: {image_path}")
        h, w = img.shape[:2]

        preprocessed = image_io.preprocess(image_path)
        lattice = self._detector(preprocessed)
        edges = image_io.trace_path(preprocessed, lattice)

        graph = nx.MultiGraph()
        graph.add_nodes_from(lattice.lattice_coords)
        for a, b in edges:
            graph.add_edge(a, b)

        dots_original = _undo_deskew(lattice.pixel_positions, preprocessed.rotation_deg, w, h)
        elapsed_ms = (time.monotonic() - t0) * 1000

        return DetectionResult(
            detector=self.name, model_version=self.model_version, dots=dots_original,
            width=w, height=h, processing_ms=elapsed_ms, graph=graph,
        )


_CLASSICAL = ClassicalDetector()
_ML = MLDetector()
_ML_GATED = GatedMLDetector()


def get_detector(name: str):
    if not isinstance(name, str):
        raise ValueError(f"unknown detector {name!r}; must be 'classical', 'ml', or 'ml-gated'")
    normalized = name.strip().lower().replace("_", "-")
    if normalized == "classical":
        return _CLASSICAL
    if normalized == "ml":
        return _ML
    if normalized in {"ml-gated", "gated-ml"}:
        return _ML_GATED
    raise ValueError(f"unknown detector {name!r}; must be 'classical', 'ml', or 'ml-gated'")

