from dataclasses import dataclass
from typing import List, Dict
import numpy as np
from collections import defaultdict
from evaluator import EvaluationResult

@dataclass
class MetricsSummary:
    condition: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    fpr: float
    fnr: float
    avg_latency: float
    enhancement_rate: float
    total_samples: int

class MetricsCalculator:
    def __init__(self):
        pass

    def compute(self, results: List[EvaluationResult]) -> Dict[str, MetricsSummary]:
        grouped = defaultdict(list)
        for r in results:
            grouped[r.condition].append(r)
        
        summaries = {}
        for condition, res_list in grouped.items():
            summaries[condition] = self._compute_single_condition(condition, res_list)
        
        # Overall
        summaries['Overall'] = self._compute_single_condition('Overall', results)
        
        return summaries

    def _compute_single_condition(self, condition: str, results: List[EvaluationResult]) -> MetricsSummary:
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        
        total_latency = 0
        enhancement_count = 0
        total_images = len(results)
        
        for res in results:
            total_latency += res.latency
            if res.enhancement_used:
                enhancement_count += 1
            
            # If no faces detected
            if res.num_detected == 0:
                if res.expected_identity:
                    # Expected someone, got nothing -> FN
                    fn += 1
                else:
                    # Expected Unknown/Nobody (if strictly testing for valid faces), and got nothing
                    # If "unknown_faces" dataset means "Faces of unknown people", then detection is expected, but identity is unknown.
                    # If it means "No faces", then TN.
                    # Usually "unknown_faces" means "Impostors" -> We expect detection, but result should be Unknown.
                    # If 0 detections, then it's a failure to detect the face at all.
                    # Assuming input images ALWAYS contain a face for this benchmark.
                    # So 0 detection is almost always FN for detection.
                    # But for "Recognition Accuracy", if we can't detect, we can't recognize.
                    fn += 1
            else:
                # We have detections
                for m in res.matches:
                    status = m.get('status')
                    if status == 'Correct': 
                        tp += 1
                    elif status == 'CorrectReject':
                        tn += 1
                    elif status == 'Missed':
                        fn += 1
                    elif status == 'FalseIdentification':
                        fp += 1
                    elif status == 'FalsePositive_UnknownMatched':
                        fp += 1

        # Accuracy = (TP + TN) / Total Matches?
        # Or Total Images?
        # Biometrics often use Transactions.
        # Here we sum up counters across all samples.
        
        total_predictions = tp + tn + fp + fn
        if total_predictions == 0:
            accuracy = 0
            precision = 0
            recall = 0
            f1 = 0
            fpr = 0
            fnr = 0
        else:
            accuracy = (tp + tn) / total_predictions
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # FPR: False Positive / (False Positive + True Negative)
            # Measures: Of all negatives (impostors), how many were accepted?
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            # FNR: False Negative / (False Negative + True Positive)
            # Measures: Of all positives (genuine), how many were rejected?
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        avg_lat = total_latency / total_images if total_images > 0 else 0
        enh_rate = enhancement_count / total_images if total_images > 0 else 0
        
        return MetricsSummary(
            condition=condition,
            accuracy=round(float(accuracy), 4),
            precision=round(float(precision), 4),
            recall=round(float(recall), 4),
            f1_score=round(float(f1), 4),
            fpr=round(float(fpr), 4),
            fnr=round(float(fnr), 4),
            avg_latency=round(avg_lat, 4),
            enhancement_rate=round(enh_rate, 4),
            total_samples=total_images
        )
