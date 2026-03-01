import matplotlib.pyplot as plt
import os
import numpy as np
from typing import Dict
from metrics import MetricsSummary

class Visualizer:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_metrics(self, summaries: Dict[str, MetricsSummary]):
        conditions = [k for k in summaries.keys() if k != 'Overall']
        if not conditions:
            return

        # 1. Accuracy vs Condition
        plt.figure(figsize=(10, 6))
        accuracies = [summaries[c].accuracy for c in conditions]
        plt.bar(conditions, accuracies, color='skyblue')
        plt.title('Accuracy by Condition')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'accuracy_by_condition.png'))
        plt.close()

        # 2. Latency vs Condition
        plt.figure(figsize=(10, 6))
        latencies = [summaries[c].avg_latency for c in conditions]
        plt.bar(conditions, latencies, color='salmon')
        plt.title('Average Latency by Condition')
        plt.ylabel('Latency (s)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'latency_by_condition.png'))
        plt.close()

        # 3. FPR vs FNR
        plt.figure(figsize=(10, 6))
        fprs = [summaries[c].fpr for c in conditions]
        fnrs = [summaries[c].fnr for c in conditions]
        x = np.arange(len(conditions))
        width = 0.35
        plt.bar(x - width/2, fprs, width, label='FPR')
        plt.bar(x + width/2, fnrs, width, label='FNR')
        plt.title('False Positive vs False Negative Rates')
        plt.ylabel('Rate')
        plt.xticks(x, conditions, rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'fpr_fnr_comparison.png'))
        plt.close()
        
        # 4. Enhancement Usage
        plt.figure(figsize=(10, 6))
        enh_rates = [summaries[c].enhancement_rate for c in conditions]
        plt.bar(conditions, enh_rates, color='lightgreen')
        plt.title('Enhancement Usage Rate by Condition')
        plt.ylabel('Usage Rate')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'enhancement_usage.png'))
        plt.close()

    def plot_confusion_matrix(self, matrix: np.ndarray, labels: list, title: str, filename: str):
        plt.figure(figsize=(10, 8))
        plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels, rotation=45)
        plt.yticks(tick_marks, labels)

        # Annotate
        thresh = matrix.max() / 2.
        for i, j in np.ndindex(matrix.shape):
            plt.text(j, i, format(matrix[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if matrix[i, j] > thresh else "black")

        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()
