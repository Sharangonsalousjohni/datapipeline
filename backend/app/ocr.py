"""Configurable OCR providers.

Providers:
* ``rapidocr``  - PaddleOCR detection/recognition models executed with ONNX Runtime
                  (pip-installable, models bundled, Apache-2.0). Default in ``auto`` mode.
* ``paddleocr`` - the original PaddleOCR package (optional, heavy dependency).
* ``tesseract`` - Tesseract via pytesseract (requires the tesseract binary).
* ``none``      - OCR disabled; scanned content is reported as an extraction warning.

All providers return lines with bounding boxes in the coordinate system of the
image that was passed in, and a confidence in [0, 1].
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Protocol

import numpy as np
from PIL import Image

from app.config import get_settings

log = logging.getLogger(__name__)


@dataclass
class OcrLine:
    text: str
    bbox: list[float]  # x0, y0, x1, y1
    confidence: float


@dataclass
class OcrResult:
    provider: str
    lines: list[OcrLine] = field(default_factory=list)
    width: int = 0
    height: int = 0
    error: str | None = None

    @property
    def mean_confidence(self) -> float | None:
        if not self.lines:
            return None
        total = sum(len(l.text) for l in self.lines) or 1
        return float(sum(l.confidence * len(l.text) for l in self.lines) / total)


class OcrProvider(Protocol):
    name: str

    def recognize(self, image: Image.Image) -> list[OcrLine]: ...


class OcrUnavailable(RuntimeError):
    pass


# --------------------------------------------------------------------------- providers
class RapidOcrProvider:
    name = "rapidocr"

    def __init__(self) -> None:
        try:
            from rapidocr_onnxruntime import RapidOCR  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on env
            raise OcrUnavailable(f"rapidocr_onnxruntime not installed: {exc}") from exc
        self._engine = RapidOCR()
        self._lock = threading.Lock()

    def recognize(self, image: Image.Image) -> list[OcrLine]:
        arr = np.array(image.convert("RGB"))[:, :, ::-1]  # BGR
        with self._lock:
            result, _ = self._engine(arr)
        lines: list[OcrLine] = []
        for box, text, score in result or []:
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            lines.append(OcrLine(str(text), [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))], float(score)))
        return lines


class PaddleOcrProvider:
    name = "paddleocr"

    def __init__(self) -> None:
        try:
            from paddleocr import PaddleOCR  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise OcrUnavailable(f"paddleocr not installed: {exc}") from exc
        self._engine = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
        self._lock = threading.Lock()

    def recognize(self, image: Image.Image) -> list[OcrLine]:  # pragma: no cover - optional dep
        arr = np.array(image.convert("RGB"))[:, :, ::-1]
        with self._lock:
            result = self._engine.ocr(arr, cls=True)
        lines: list[OcrLine] = []
        for page in result or []:
            for box, (text, score) in page or []:
                xs = [p[0] for p in box]
                ys = [p[1] for p in box]
                lines.append(OcrLine(str(text), [min(xs), min(ys), max(xs), max(ys)], float(score)))
        return lines


class TesseractProvider:
    name = "tesseract"

    def __init__(self) -> None:
        try:
            import pytesseract  # type: ignore

            pytesseract.get_tesseract_version()
        except Exception as exc:
            raise OcrUnavailable(f"tesseract unavailable: {exc}") from exc
        self._pt = pytesseract

    def recognize(self, image: Image.Image) -> list[OcrLine]:
        settings = get_settings()
        data = self._pt.image_to_data(
            image.convert("L"), lang=settings.ocr_languages, output_type=self._pt.Output.DICT
        )
        groups: dict[tuple[int, int, int], list[int]] = {}
        for i, word in enumerate(data["text"]):
            if not str(word).strip():
                continue
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            groups.setdefault(key, []).append(i)
        lines: list[OcrLine] = []
        for idxs in groups.values():
            words = [str(data["text"][i]) for i in idxs]
            confs = [float(data["conf"][i]) for i in idxs if float(data["conf"][i]) >= 0]
            x0 = min(data["left"][i] for i in idxs)
            y0 = min(data["top"][i] for i in idxs)
            x1 = max(data["left"][i] + data["width"][i] for i in idxs)
            y1 = max(data["top"][i] + data["height"][i] for i in idxs)
            conf = (sum(confs) / len(confs) / 100.0) if confs else 0.0
            lines.append(OcrLine(" ".join(words), [float(x0), float(y0), float(x1), float(y1)], conf))
        lines.sort(key=lambda l: (l.bbox[1], l.bbox[0]))
        return lines


_PROVIDERS = {"rapidocr": RapidOcrProvider, "paddleocr": PaddleOcrProvider, "tesseract": TesseractProvider}
_provider_lock = threading.Lock()


@lru_cache(maxsize=1)
def _load_provider(name: str) -> OcrProvider | None:
    order = ["rapidocr", "tesseract", "paddleocr"] if name == "auto" else [name]
    if name == "none":
        return None
    for candidate in order:
        try:
            provider = _PROVIDERS[candidate]()
            log.info("ocr_provider_loaded", extra={"provider": candidate})
            return provider
        except OcrUnavailable as exc:
            log.warning("ocr_provider_unavailable", extra={"provider": candidate, "reason": str(exc)})
    return None


def get_ocr_provider() -> OcrProvider | None:
    with _provider_lock:
        return _load_provider(get_settings().ocr_provider)


def ocr_available() -> bool:
    return get_ocr_provider() is not None


# --------------------------------------------------------------------------- preprocessing
def _deskew_angle(gray: np.ndarray) -> float:
    import cv2

    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 50:
        return 0.0
    angle = cv2.minAreaRect(coords.astype(np.float32))[-1]
    # OpenCV >= 4.5 returns angle in [0, 90); map to [-45, 45]
    if angle > 45:
        angle -= 90
    return float(-angle) if abs(angle) < 45 else 0.0


def preprocess(image: Image.Image) -> tuple[Image.Image, np.ndarray | None, float]:
    """Grayscale, upscale small images and deskew.

    Returns the processed image, the inverse affine matrix (to map boxes back to
    the original image coordinates) and the scale factor applied.
    """
    import cv2

    gray = np.array(image.convert("L"))
    scale = 1.0
    h, w = gray.shape
    if max(h, w) < 1000:
        scale = 1000.0 / max(h, w)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    angle = _deskew_angle(gray)
    inverse = None
    if 0.4 < abs(angle) < 10:
        h, w = gray.shape
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        gray = cv2.warpAffine(gray, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        inverse = cv2.invertAffineTransform(m)
    return Image.fromarray(gray), inverse, scale


def _map_box(box: list[float], inverse: np.ndarray | None, scale: float) -> list[float]:
    x0, y0, x1, y1 = box
    pts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)
    if inverse is not None:
        pts = np.hstack([pts, np.ones((4, 1))]) @ inverse.T
    pts = pts / scale
    return [round(float(pts[:, 0].min()), 1), round(float(pts[:, 1].min()), 1),
            round(float(pts[:, 0].max()), 1), round(float(pts[:, 1].max()), 1)]


def run_ocr(image: Image.Image) -> OcrResult:
    """OCR an image, returning boxes in the *original* image coordinates."""
    provider = get_ocr_provider()
    width, height = image.size
    if provider is None:
        return OcrResult(provider="none", width=width, height=height, error="No OCR provider available")
    settings = get_settings()
    try:
        if settings.ocr_preprocess:
            processed, inverse, scale = preprocess(image)
        else:
            processed, inverse, scale = image, None, 1.0
        lines = provider.recognize(processed)
        for line in lines:
            line.bbox = _map_box(line.bbox, inverse, scale)
        lines = [l for l in lines if l.text.strip()]
        return OcrResult(provider=provider.name, lines=lines, width=width, height=height)
    except Exception as exc:  # OCR failure must not crash the pipeline
        log.exception("ocr_failed", extra={"provider": provider.name})
        return OcrResult(provider=provider.name, width=width, height=height, error=f"OCR failed: {type(exc).__name__}")


def group_lines_into_paragraphs(lines: list[OcrLine]) -> list[list[OcrLine]]:
    """Group OCR lines into paragraphs using vertical gaps between lines."""
    if not lines:
        return []
    ordered = sorted(lines, key=lambda l: (round(l.bbox[1] / 5), l.bbox[0]))
    heights = sorted(max(1.0, l.bbox[3] - l.bbox[1]) for l in ordered)
    median_h = heights[len(heights) // 2]
    paragraphs: list[list[OcrLine]] = [[ordered[0]]]
    for prev, cur in zip(ordered, ordered[1:]):
        gap = cur.bbox[1] - prev.bbox[3]
        if gap > 0.9 * median_h:
            paragraphs.append([cur])
        else:
            paragraphs[-1].append(cur)
    return paragraphs
