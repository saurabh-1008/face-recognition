"""Shared CPU face extraction and model-specific JSON storage."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / ".venv" / "insightface"
DATABASE = ROOT / "data" / "embeddings" / "users.json"
MODELS = ("buffalo_l", "buffalo_m", "buffalo_s", "buffalo_sc")


def load_model(name):
    from insightface.app import FaceAnalysis
    from insightface.utils import ensure_available
    pack = Path(ensure_available("models", name, root=str(MODEL_ROOT)))
    # Some official archives wrap ONNX files in an extra pack-name directory.
    nested = pack / name
    if not list(pack.glob("*.onnx")) and nested.is_dir():
        for file in nested.iterdir():
            if file.is_file() and not (pack / file.name).exists():
                file.replace(pack / file.name)
    app = FaceAnalysis(name=name, root=str(MODEL_ROOT),
                       allowed_modules=["detection", "recognition"],
                       providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    return app


def normalize(vector):
    vector = np.asarray(vector, dtype=np.float32)
    if vector.shape != (512,) or not np.isfinite(vector).all():
        raise ValueError("Expected 512 finite embedding values")
    length = np.linalg.norm(vector)
    if length <= 0:
        raise ValueError("Embedding has zero norm")
    return vector / length


def extract(app, path):
    import cv2
    image = cv2.imread(str(Path(path).resolve()))
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    faces = app.get(image)
    if len(faces) != 1:
        raise ValueError(f"Expected exactly one face in {path}; found {len(faces)}")
    return normalize(faces[0].embedding)


def cosine(first, second):
    return float(np.clip(np.dot(normalize(first), normalize(second)), -1, 1))


def read_database():
    if not DATABASE.exists():
        return {"schema_version": 1, "users": {}}
    data = json.loads(DATABASE.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or not isinstance(data.get("users"), dict):
        raise ValueError("Unsupported users.json schema")
    return data


def save_database(data):
    DATABASE.parent.mkdir(parents=True, exist_ok=True)
    temporary = DATABASE.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(DATABASE)
