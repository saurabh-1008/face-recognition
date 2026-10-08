import cv2
import numpy as np
from pathlib import Path
from insightface.app import FaceAnalysis

# Load InsightFace
app = FaceAnalysis(name="buffalo_s")
app.prepare(ctx_id=0)

# Load new/test image
image_path = Path(__file__).with_name("IMG_20251213_193020.jpg")
image = cv2.imread(str(image_path))
if image is None:
    raise FileNotFoundError(f"Could not read image: {image_path}")
faces = app.get(image)

if len(faces) != 1:
    print("Expected exactly one face")
    exit()

# Generate embedding for new face
new_embedding = faces[0].embedding

# Load registered user's embedding
stored_embedding = np.load("face-embeddings/embeddings/user_001.npy")


# Cosine similarity
similarity = np.dot(new_embedding, stored_embedding) / (
    np.linalg.norm(new_embedding) * np.linalg.norm(stored_embedding)
)

print("Similarity:", similarity)


# Temporary threshold
THRESHOLD = 0.5

if similarity >= THRESHOLD:
    print("MATCH - User authenticated")
else:
    print("NO MATCH - Unknown user")