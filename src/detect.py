try:
    from insightface.app import FaceAnalysis
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "InsightFace is not installed. Please install it using:\n"
        "pip install insightface onnxruntime\n"
        "This project cannot run without InsightFace."
    ) from e

class FaceDetector:
    def __init__(self, model_path):
        self.app = FaceAnalysis(
            name='buffalo_l',
            root=model_path,
            providers=['CPUExecutionProvider']
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def detect(self, image):
        if image is None:
            raise ValueError("Input image is None. Check image path.")
        return self.app.get(image)
