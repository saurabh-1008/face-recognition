"""MiniFASNet ONNX liveness gate used before InsightFace verification.

The preprocessing and class order follow facenox/face-antispoof-onnx:
128x128 RGB input in [0, 1], with output logits [real, spoof].
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LIVENESS_MODEL = ROOT / "models" / "minifasnet" / "best_model_quantized.onnx"
DETECTOR_MODEL = ROOT / "models" / "minifasnet" / "detector_quantized.onnx"
MODEL_IMAGE_SIZE = 128


class LivenessError(ValueError):
    """Raised when liveness cannot be evaluated safely."""


@dataclass(frozen=True)
class LivenessResult:
    is_real: bool
    real_probability: float
    real_logit: float
    spoof_logit: float


def _read_rgb(source, cv2_module):
    if isinstance(source, np.ndarray):
        image = source
        source_name = "webcam frame"
    else:
        image = cv2_module.imread(str(Path(source).resolve()))
        source_name = str(source)
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise LivenessError(f"Cannot read a three-channel image from {source_name}")
    return cv2_module.cvtColor(image, cv2_module.COLOR_BGR2RGB), source_name


def _load_detector(model_path, cv2_module):
    path = Path(model_path)
    if not path.is_file():
        raise LivenessError(f"YuNet face detector is missing: {path}")
    try:
        return cv2_module.FaceDetectorYN.create(
            str(path), "", (320, 320), 0.8, 0.3, 5000
        )
    except Exception as error:
        raise LivenessError(f"Could not load YuNet face detector: {error}") from error


def _detect_one_face(
    image, detector, cv2_module, margin=5, min_face_size=60, max_dimension=640,
    return_landmarks=False,
):
    height, width = image.shape[:2]
    scale = min(1.0, max_dimension / max(height, width))
    if scale < 1.0:
        detector_image = cv2_module.resize(
            image,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2_module.INTER_AREA,
        )
    else:
        detector_image = image
    detector_height, detector_width = detector_image.shape[:2]
    detector.setInputSize((detector_width, detector_height))
    try:
        _, faces = detector.detect(detector_image)
    except Exception as error:
        raise LivenessError(f"MiniFASNet face detection failed: {error}") from error

    valid = []
    landmarks = []
    for face in [] if faces is None else faces:
        x, y, face_width, face_height = (face[:4] / scale).astype(int)
        inside = (
            x >= margin
            and y >= margin
            and x + face_width <= width - margin
            and y + face_height <= height - margin
        )
        if inside and face_width >= min_face_size and face_height >= min_face_size:
            valid.append((x, y, x + face_width, y + face_height))
            if return_landmarks:
                landmarks.append(np.asarray(face[4:14], dtype=np.float32).reshape(5, 2) / scale)
    if len(valid) != 1:
        raise LivenessError(
            f"Expected exactly one complete face for liveness; found {len(valid)}"
        )
    return (valid[0], landmarks[0]) if return_landmarks else valid[0]


def _expanded_square_crop(image, bbox, cv2_module, expansion=1.5):
    image_height, image_width = image.shape[:2]
    x1, y1, x2, y2 = bbox
    side = max(x2 - x1, y2 - y1)
    crop_size = max(1, int(side * expansion))
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    left = int(center_x - crop_size / 2)
    top = int(center_y - crop_size / 2)
    right = left + crop_size
    bottom = top + crop_size

    clipped = image[max(0, top):min(image_height, bottom),
                    max(0, left):min(image_width, right)]
    if clipped.size == 0:
        raise LivenessError("The detected face crop is empty")
    result = cv2_module.copyMakeBorder(
        clipped,
        max(0, -top),
        max(0, bottom - image_height),
        max(0, -left),
        max(0, right - image_width),
        cv2_module.BORDER_REFLECT_101,
    )
    if result.shape[:2] != (crop_size, crop_size):
        raise LivenessError("Could not create a valid square face crop")
    return result


def _preprocess(face_crop, cv2_module, size=MODEL_IMAGE_SIZE):
    interpolation = (
        cv2_module.INTER_LANCZOS4
        if max(face_crop.shape[:2]) < size
        else cv2_module.INTER_AREA
    )
    resized = cv2_module.resize(face_crop, (size, size), interpolation=interpolation)
    return np.ascontiguousarray(
        resized.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    )


def _load_liveness_session(model_path, ort_module):
    path = Path(model_path)
    if not path.is_file():
        raise LivenessError(f"ONNX liveness model is missing: {path}")
    try:
        options = ort_module.SessionOptions()
        options.graph_optimization_level = ort_module.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.execution_mode = ort_module.ExecutionMode.ORT_SEQUENTIAL
        session = ort_module.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )
    except Exception as error:
        raise LivenessError(f"Could not load ONNX liveness model: {error}") from error
    return session


def check_liveness(
    source,
    threshold=0.5,
    liveness_model=LIVENESS_MODEL,
    detector_model=DETECTOR_MODEL,
    cv2_module=None,
    ort_module=None,
    detector=None,
    session=None,
):
    """Return a liveness decision for one face, failing closed on any error."""
    if not 0 < threshold < 1:
        raise LivenessError("Liveness threshold must be greater than 0 and less than 1")
    if cv2_module is None:
        import cv2 as cv2_module
    if ort_module is None:
        import onnxruntime as ort_module

    image, _ = _read_rgb(source, cv2_module)
    if detector is None:
        detector = _load_detector(detector_model, cv2_module)
    bbox = _detect_one_face(image, detector, cv2_module)
    crop = _expanded_square_crop(image, bbox, cv2_module)
    tensor = _preprocess(crop, cv2_module)
    if session is None:
        session = _load_liveness_session(liveness_model, ort_module)

    try:
        input_name = session.get_inputs()[0].name
        logits = np.asarray(session.run([], {input_name: tensor})[0], dtype=np.float32)
    except Exception as error:
        raise LivenessError(f"MiniFASNet inference failed: {error}") from error
    if logits.shape != (1, 2) or not np.isfinite(logits).all():
        raise LivenessError(f"Unexpected MiniFASNet output shape or values: {logits.shape}")

    real_logit, spoof_logit = map(float, logits[0])
    difference = real_logit - spoof_logit
    real_probability = float(1.0 / (1.0 + np.exp(-np.clip(difference, -80, 80))))
    return LivenessResult(
        is_real=real_probability >= threshold,
        real_probability=real_probability,
        real_logit=real_logit,
        spoof_logit=spoof_logit,
    )
