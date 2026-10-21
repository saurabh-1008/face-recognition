"""Model registry and CPU ONNX adapters for the shared liveness experiment."""
import argparse
from dataclasses import dataclass
import hashlib
from pathlib import Path
import urllib.request

import numpy as np

from anti_spoof import (
    ROOT, DETECTOR_MODEL, LIVENESS_MODEL, LivenessError, check_liveness,
    _detect_one_face, _load_detector, _load_liveness_session, _read_rgb,
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    title: str
    path: Path
    size: int
    threshold: float
    output_kind: str
    crop: str
    source: str
    sha256: str
    download_url: str


TINY_REVISION = "8d92198e7ee122f39b2585e691a3d9450ca0cb0f"
MODEL_SPECS = {
    "minifasnet": ModelSpec(
        "minifasnet", "MiniFASNet V2 SE (current baseline)", LIVENESS_MODEL, 128, 0.5,
        "two_logits", "1.5x square; legacy RGB detector input",
        "https://github.com/facenox/face-antispoof-onnx",
        "fde20585635cae62ed1d41796f76b6f8bc4b92cd91ec1cf0f1bc6485d2d587a9",
        "https://raw.githubusercontent.com/facenox/face-antispoof-onnx/main/models/best_model_quantized.onnx"),
    "anti-spoof-mn3": ModelSpec(
        "anti-spoof-mn3", "MobileNetV3 / CelebA-Spoof", ROOT / "models/anti-spoof-mn3/anti-spoof-mn3.onnx",
        128, 0.5, "two_probabilities", "bounding box expanded 1.1x horizontally, 1.05x vertically; BGR detector input",
        "https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/anti-spoof-mn3",
        "c4c99af04603b62d7e44f6f4daeb33e0daeccc696008c0b1d62f6f5cebbb3262",
        "https://storage.openvinotoolkit.org/repositories/open_model_zoo/public/2022.1/anti-spoof-mn3/anti-spoof-mn3.onnx"),
    "tinyliveness": ModelSpec(
        "tinyliveness", "TinyLiveness EfficientNet-B0", ROOT / "models/tinyliveness/tinyliveness_main_apcer1_224.onnx",
        224, 0.99, "live_probability", "YuNet five-point similarity alignment to 224px ArcFace template; BGR detector input",
        f"https://github.com/yuvrajraina/TinyLiveness/tree/{TINY_REVISION}",
        "fbf9258f27b6b597c8fc8767a1add930628fe90b396126b15171b7f54985c670",
        f"https://raw.githubusercontent.com/yuvrajraina/TinyLiveness/{TINY_REVISION}/checkpoints/tinyliveness_main_apcer1_224.onnx"),
}


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def list_models():
    for name, spec in MODEL_SPECS.items():
        status = "installed" if spec.path.is_file() else "not installed"
        print(f"{name:18} {status:14} threshold={spec.threshold}  {spec.title}")


def download_model(name):
    spec = MODEL_SPECS[name]
    if spec.path.exists():
        if file_hash(spec.path) != spec.sha256:
            raise LivenessError(f"Existing model differs from the registered release: {spec.path}")
        print(f"Already installed: {name}")
        return
    with urllib.request.urlopen(spec.download_url, timeout=90) as response:
        data = response.read(50_000_001)
    if len(data) > 50_000_000 or hashlib.sha256(data).hexdigest() != spec.sha256:
        raise LivenessError(f"Download checksum mismatch for {name}; nothing installed")
    spec.path.parent.mkdir(parents=True, exist_ok=True)
    with spec.path.open("xb") as stream:
        stream.write(data)
    print(f"Installed: {name}")


def preprocess(rgb_crop, spec, cv2):
    interpolation = cv2.INTER_CUBIC if spec.name == "anti-spoof-mn3" else cv2.INTER_LINEAR
    pixels = cv2.resize(rgb_crop, (spec.size, spec.size), interpolation=interpolation).astype(np.float32)
    if spec.name == "anti-spoof-mn3":
        pixels = (pixels - np.array([151.2405, 119.595, 107.8395], dtype=np.float32)) / np.array(
            [63.0105, 56.457, 55.0035], dtype=np.float32)
    elif spec.name == "tinyliveness":
        pixels = (pixels / 255.0 - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
            [0.229, 0.224, 0.225], dtype=np.float32)
    else:
        raise LivenessError("MiniFASNet uses its existing baseline preprocessing")
    return np.ascontiguousarray(pixels.transpose(2, 0, 1)[None])


def probability_from_output(output, spec):
    values = np.asarray(output, dtype=np.float32)
    expected = (1, 1) if spec.output_kind == "live_probability" else (1, 2)
    if values.shape != expected or not np.isfinite(values).all():
        raise LivenessError(f"Unexpected {spec.name} output: {values.shape}; expected {expected} finite values")
    if spec.output_kind == "two_logits":
        return float(1 / (1 + np.exp(-np.clip(float(values[0, 0] - values[0, 1]), -80, 80))))
    if np.any(values < 0) or np.any(values > 1):
        raise LivenessError(f"{spec.name} returned values outside [0,1]")
    if spec.output_kind == "two_probabilities" and not np.isclose(values.sum(), 1.0, atol=1e-4):
        raise LivenessError(f"{spec.name} output probabilities do not sum to 1")
    return float(values[0, 0])


def align_face(rgb, landmarks, cv2):
    """Deterministic least-squares similarity transform; alignment is our adapter choice."""
    target = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                       [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32) * 2
    points = np.asarray(landmarks, dtype=np.float32)
    if points.shape != (5, 2) or not np.isfinite(points).all():
        raise LivenessError("Invalid landmarks for TinyLiveness alignment")
    matrix = np.zeros((10, 4), dtype=np.float64)
    matrix[0::2, 0] = matrix[1::2, 1] = points[:, 0]
    matrix[0::2, 1] = -points[:, 1]
    matrix[1::2, 0] = points[:, 1]
    matrix[0::2, 2] = matrix[1::2, 3] = 1
    coefficients, _, rank, _ = np.linalg.lstsq(matrix, target.reshape(-1), rcond=None)
    a, b, tx, ty = coefficients
    if rank < 4 or not np.isfinite(coefficients).all() or a * a + b * b < 1e-8:
        raise LivenessError("Cannot align degenerate face landmarks")
    transform = np.array([[a, -b, tx], [b, a, ty]], dtype=np.float32)
    return cv2.warpAffine(rgb, transform, (224, 224), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT_101)


@dataclass(frozen=True)
class Prediction:
    real_probability: float
    is_real: bool


class LivenessEngine:
    def __init__(self, name="minifasnet", threshold=None, model_path=None,
                 detector_path=DETECTOR_MODEL):
        import cv2
        import onnxruntime as ort
        self.spec = MODEL_SPECS[name]
        self.threshold = self.spec.threshold if threshold is None else threshold
        if not 0 < self.threshold < 1:
            raise LivenessError("Threshold must be finite and between 0 and 1")
        self.path = Path(model_path or self.spec.path).resolve()
        self.detector_path = Path(detector_path).resolve()
        if not self.path.is_file():
            raise LivenessError(f"Model missing: {self.path}. Install with: python scripts/liveness_models.py --download {name}")
        self.cv2 = cv2
        self.ort = ort
        self.detector = _load_detector(self.detector_path, cv2)
        self.session = _load_liveness_session(self.path, ort)
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise LivenessError(f"{name} requires one model input and output")
        shape = inputs[0].shape
        if inputs[0].type != "tensor(float)" or len(shape) != 4 or list(shape[1:]) != [3, self.spec.size, self.spec.size]:
            raise LivenessError(f"Model does not match {name}'s input contract: {shape}")
        if isinstance(shape[0], int) and shape[0] != 1:
            raise LivenessError("Model must accept a single-frame batch")
        self.input_name = inputs[0].name
        self.output_name = outputs[0].name
        self.metadata = {
            "name": name, "title": self.spec.title, "path": str(self.path),
            "sha256": file_hash(self.path), "source": self.spec.source,
            "threshold": self.threshold, "default_threshold": self.spec.threshold,
            "output_kind": self.spec.output_kind, "crop": self.spec.crop,
            "detector_path": str(self.detector_path), "detector_sha256": file_hash(self.detector_path),
            "input_size": self.spec.size, "provider": "CPUExecutionProvider",
            "adapter_version": 1,
            "opencv_version": cv2.__version__, "onnxruntime_version": ort.__version__,
        }
        self.metadata["registered_release"] = self.metadata["sha256"] == self.spec.sha256

    def predict(self, source):
        if self.spec.name == "minifasnet":
            return check_liveness(source, threshold=self.threshold, cv2_module=self.cv2,
                                  ort_module=self.ort, detector=self.detector, session=self.session)
        rgb, _ = _read_rgb(source, self.cv2)
        bgr = self.cv2.cvtColor(rgb, self.cv2.COLOR_RGB2BGR)
        bbox, landmarks = _detect_one_face(bgr, self.detector, self.cv2, return_landmarks=True)
        if self.spec.name == "tinyliveness":
            crop = align_face(rgb, landmarks, self.cv2)
        else:
            x1, y1, x2, y2 = bbox
            dx, dy = (x2 - x1) * 0.05, (y2 - y1) * 0.025
            crop = rgb[max(0, int(y1 - dy)):min(rgb.shape[0], int(y2 + dy)),
                       max(0, int(x1 - dx)):min(rgb.shape[1], int(x2 + dx))]
        tensor = preprocess(crop, self.spec, self.cv2)
        try:
            output = self.session.run([self.output_name], {self.input_name: tensor})[0]
        except Exception as error:
            raise LivenessError(f"{self.spec.name} inference failed: {error}") from error
        probability = probability_from_output(output, self.spec)
        return Prediction(probability, probability >= self.threshold)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--list", action="store_true")
    action.add_argument("--download", choices=[*MODEL_SPECS, "all"])
    args = parser.parse_args()
    try:
        if args.list:
            list_models()
        else:
            for name in MODEL_SPECS if args.download == "all" else [args.download]:
                download_model(name)
    except (ValueError, OSError) as error:
        parser.exit(2, f"Model setup failed: {error}\n")


if __name__ == "__main__":
    main()
