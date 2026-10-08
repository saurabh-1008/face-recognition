import cv2
from pathlib import Path
import numpy as np
from insightface.app import FaceAnalysis

app=FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0)
image_path = Path(__file__).with_name("IMG_20251213_193020.jpg")
image = cv2.imread(str(image_path))
if image is None:
    raise FileNotFoundError(f"Could not read image: {image_path}")

faces=app.get(image)
print("face detected:", len(faces))
if len(faces) == 1:
    embedding = faces[0].embedding

    print("Embedding shape:", embedding.shape)
    print("Embedding:", embedding)
else:
    print("Expected exactly one face")

user_id = "user_001"

np.save(f"embeddings/{user_id}.npy", embedding)

print("User registered successfully")
print("User ID:", user_id)
print("Embedding shape:", embedding.shape)