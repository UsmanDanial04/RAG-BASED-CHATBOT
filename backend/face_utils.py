"""
face_utils.py

Two-stage pipeline:
  1. MediaPipe Face Detection -> finds the face, checks quality
     (single face, big enough, roughly centered) before we bother
     encoding it. Gives real-time UX feedback ("no face detected",
     "move closer", "multiple faces found").

  2. InsightFace (ArcFace via ONNX Runtime) -> turns the cropped
     face into a 512-d embedding vector we can store and compare
     with cosine distance. No dlib / no C++ compilation required —
     works on Python 3.13+ out of the box.

NOTE: InsightFace downloads model weights (~200 MB) to ~/.insightface/
on first run. Make sure the server has internet access the first time.
"""

import base64
import io
import warnings
from dataclasses import dataclass, field

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

warnings.filterwarnings("ignore")  # suppress verbose onnxruntime warnings

mp_face_detection = mp.solutions.face_detection

# ---------------------------------------------------------------------------
# InsightFace setup — lazy-loaded so import errors are caught cleanly
# ---------------------------------------------------------------------------
_insight_app = None


def _get_insight_app():
    """Lazy-load InsightFace FaceAnalysis app (downloads models on first call)."""
    global _insight_app
    if _insight_app is None:
        try:
            from insightface.app import FaceAnalysis  # noqa: PLC0415
            _insight_app = FaceAnalysis(
                name="buffalo_sc",          # lightweight: detector + ArcFace
                providers=["CPUExecutionProvider"],
            )
            _insight_app.prepare(ctx_id=-1, det_size=(320, 320))
        except Exception as exc:
            raise RuntimeError(
                "Could not load InsightFace models. "
                "Run: pip install insightface onnxruntime\n"
                f"Original error: {exc}"
            ) from exc
    return _insight_app


# ---------------------------------------------------------------------------
# Tunable parameters
# ---------------------------------------------------------------------------
# Cosine DISTANCE (0=identical, 2=opposite).
# Lower threshold = stricter (fewer false accepts, more false rejects).
# 0.40 is a good starting point; raise to ~0.50 if you get too many rejects.
MATCH_THRESHOLD = 0.40


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------
@dataclass
class FaceCheckResult:
    ok: bool
    message: str
    face_image: np.ndarray | None = None   # cropped BGR face, only set if ok


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def decode_base64_image(data_url: str) -> np.ndarray:
    """
    Convert a 'data:image/jpeg;base64,...' string from the browser
    into a BGR numpy array.
    """
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    img_bytes = base64.b64decode(data_url)
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    rgb = np.array(pil_img)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def find_and_validate_face(bgr_image: np.ndarray) -> FaceCheckResult:
    """
    Run MediaPipe Face Detection on the frame and validate:
    - Exactly one face visible
    - Face is large enough (not too far away)
    - Face is not clipped at the edges

    Returns a FaceCheckResult with ok=True and the cropped face
    region if all checks pass.
    """
    h, w = bgr_image.shape[:2]
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    with mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.6
    ) as detector:
        results = detector.process(rgb_image)

    if not results.detections:
        return FaceCheckResult(False, "No face detected. Look at the camera.")

    if len(results.detections) > 1:
        return FaceCheckResult(False, "Multiple faces detected. Only one person at a time.")

    detection = results.detections[0]
    box = detection.location_data.relative_bounding_box

    face_area_ratio = box.width * box.height
    if face_area_ratio < 0.05:
        return FaceCheckResult(False, "Face too far from camera. Move closer.")

    x1 = max(int(box.xmin * w), 0)
    y1 = max(int(box.ymin * h), 0)
    x2 = min(int((box.xmin + box.width) * w), w)
    y2 = min(int((box.ymin + box.height) * h), h)

    if x2 <= x1 or y2 <= y1:
        return FaceCheckResult(False, "Face partially out of frame.")

    # Pad the crop so InsightFace has enough context
    pad_x = int((x2 - x1) * 0.3)
    pad_y = int((y2 - y1) * 0.3)
    x1p = max(x1 - pad_x, 0)
    y1p = max(y1 - pad_y, 0)
    x2p = min(x2 + pad_x, w)
    y2p = min(y2 + pad_y, h)

    face_crop = bgr_image[y1p:y2p, x1p:x2p]
    return FaceCheckResult(True, "Face looks good.", face_crop)


def get_face_encoding(bgr_face_image: np.ndarray) -> np.ndarray | None:
    """
    Run InsightFace ArcFace on an already-cropped face image.
    Returns a normalized 512-d embedding vector, or None if no face found.
    """
    app = _get_insight_app()

    # InsightFace works in BGR (same as OpenCV) — no conversion needed
    faces = app.get(bgr_face_image)
    if not faces:
        # Fallback: resize and try again in case the crop is too small
        resized = cv2.resize(bgr_face_image, (112, 112))
        faces = app.get(resized)

    if not faces:
        return None

    # Take the largest detected face (most likely the correct one)
    best_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    embedding = best_face.embedding

    # L2-normalize so cosine distance == 1 - dot product
    norm = np.linalg.norm(embedding)
    if norm < 1e-6:
        return None
    return embedding / norm


def match_encoding(
    candidate: np.ndarray,
    stored_encodings: list[np.ndarray],
) -> tuple[bool, float]:
    """
    Compare a candidate (L2-normalized) embedding against one or more
    stored embeddings using cosine distance.

    cosine_distance = 1 - dot(A, B)   (range 0..2, lower = more similar)
    """
    if not stored_encodings:
        return False, 2.0

    distances = [float(1.0 - np.dot(candidate, enc)) for enc in stored_encodings]
    best_distance = min(distances)
    return best_distance <= MATCH_THRESHOLD, best_distance
