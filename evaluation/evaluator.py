import sys
import os
import time
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional

# Add parent directory to path to import core modules
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from detect import FaceDetector
from enhance import FaceEnhancer
from recognize import FaceRecognizer
from utils import quality_check

# Configuration
MODEL_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models/insightface_models')
GFPGAN_MODEL = os.path.join(MODEL_ROOT, 'GFPGANv1.4.pth')
EMBEDDINGS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data/embeddings.pkl')

@dataclass
class EvaluationResult:
    image_path: str
    condition: str
    num_detected: int
    num_recognized_correctly: int
    num_false_recognitions: int
    num_unknown_matched: int
    latency: float
    enhancement_used: bool
    matches: List[Dict] = field(default_factory=list)
    failure_reason: Optional[str] = None
    expected_identity: Optional[str] = None

class Evaluator:
    def __init__(self):
        print("Initializing Evaluation Pipeline...")
        self.detector = FaceDetector(MODEL_ROOT)
        self.recognizer = FaceRecognizer(MODEL_ROOT, EMBEDDINGS_PATH)
        self.enhancer = FaceEnhancer(GFPGAN_MODEL)
        print("Pipeline Initialized.")

    def evaluate_image(self, image_path: str, condition: str, expected_identity: Optional[str] = None) -> EvaluationResult:
        start_time = time.time()
        
        image = cv2.imread(image_path)
        if image is None:
            return EvaluationResult(
                image_path=image_path,
                condition=condition,
                num_detected=0,
                num_recognized_correctly=0,
                num_false_recognitions=0,
                num_unknown_matched=0,
                latency=0.0,
                enhancement_used=False,
                failure_reason="ImageLoadError"
            )

        try:
            # 1. Detection
            faces = self.detector.detect(image)
            num_detected = len(faces)
            
            enhancement_used = False
            matches = []
            
            correct_count = 0
            false_count = 0
            unknown_matched_count = 0

            for face in faces:
                x1, y1, x2, y2 = face.bbox.astype(int)
                face_img = image[y1:y2, x1:x2]
                
                # Check quality and enhance if needed
                if not quality_check(face_img):
                    face_img = self.enhancer.enhance(face_img)
                    enhancement_used = True
                
                # Recognition
                embedding = face.normed_embedding
                match = self.recognizer.recognize(embedding)
                
                match_result = {
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "recognized_name": match['name'] if match else "Unknown",
                    "score": match.get('score', 0.0) if match else 0.0,
                    "expected": expected_identity
                }
                
                # Evaluation Logic
                if match:
                    # System identified a person
                    detected_name = match['name']
                    
                    if expected_identity:
                        # We expected a specific person
                        if detected_name.lower() == expected_identity.lower():
                            correct_count += 1
                            match_result['status'] = 'Correct'
                        else:
                            false_count += 1
                            match_result['status'] = 'FalseIdentification'
                    else:
                        # We expected NO ONE (Unknown/Stranger)
                        # But system identified someone
                        unknown_matched_count += 1
                        match_result['status'] = 'FalsePositive_UnknownMatched'
                else:
                    # System returned Unknown
                    if expected_identity:
                        # We expected someone, but got Unknown
                        # This works as a False Negative for recognition, 
                        # but often we track it as just "Not Recognized"
                        match_result['status'] = 'Missed'
                    else:
                        # We expected Unknown, got Unknown -> Correct
                        correct_count += 1
                        match_result['status'] = 'CorrectReject'
                
                matches.append(match_result)

            latency = time.time() - start_time

            return EvaluationResult(
                image_path=image_path,
                condition=condition,
                num_detected=num_detected,
                num_recognized_correctly=correct_count,
                num_false_recognitions=false_count,
                num_unknown_matched=unknown_matched_count,
                latency=latency,
                enhancement_used=enhancement_used,
                matches=matches,
                expected_identity=expected_identity
            )

        except Exception as e:
            return EvaluationResult(
                image_path=image_path,
                condition=condition,
                num_detected=0,
                num_recognized_correctly=0,
                num_false_recognitions=0,
                num_unknown_matched=0,
                latency=time.time() - start_time,
                enhancement_used=False,
                failure_reason=str(e),
                expected_identity=expected_identity
            )
