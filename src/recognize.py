import os
import pickle
from utils import cosine_similarity

try:
    from insightface.app import FaceAnalysis
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "InsightFace is not installed. Recognition cannot run."
    ) from e

class FaceRecognizer:
    def __init__(self, model_path, embeddings_path, threshold=0.5):
        self.threshold = threshold
        self.embeddings_path = embeddings_path
        self.app = FaceAnalysis(
            name='buffalo_l',
            root=model_path,
            providers=['CPUExecutionProvider']
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.known_embeddings = self.load_embeddings()

    def load_embeddings(self):
        if os.path.exists(self.embeddings_path):
            with open(self.embeddings_path, 'rb') as f:
                return pickle.load(f)
        print("[WARN] embeddings.pkl not found. No faces will be recognized.")
        return {}

    def recognize(self, embedding):
        best_match = None
        best_score = 0.0
        for student_id, data in self.known_embeddings.items():
            score = cosine_similarity(embedding, data['embedding'])
            if score > self.threshold and score > best_score:
                best_score = score
                best_match = data
        return best_match
