import os
import csv
import json
from typing import List, Dict
from dataclasses import asdict
from evaluator import EvaluationResult
from metrics import MetricsSummary

class Reporter:
    def __init__(self, results_dir: str, logs_dir: str):
        self.results_dir = results_dir
        self.logs_dir = logs_dir
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)

    def save_json_log(self, results: List[EvaluationResult]):
        log_path = os.path.join(self.logs_dir, 'evaluation_log.json')
        # Convert dataclasses to dicts
        serializable_results = []
        for r in results:
            r_dict = asdict(r)
            serializable_results.append(r_dict)
        
        with open(log_path, 'w') as f:
            json.dump(serializable_results, f, indent=4)

    def save_metrics_csv(self, summaries: Dict[str, MetricsSummary]):
        csv_path = os.path.join(self.results_dir, 'metrics_summary.csv')
        fieldnames = [
            'Condition', 'Accuracy', 'Precision', 'Recall', 'F1', 
            'FPR', 'FNR', 'Avg_Latency', 'Enhancement_Rate'
        ]
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(fieldnames)
            
            # Sort: Overall first, then alphabetical
            keys = sorted(summaries.keys())
            if 'Overall' in keys:
                keys.remove('Overall')
                keys.insert(0, 'Overall')
                
            for key in keys:
                s = summaries[key]
                writer.writerow([
                    s.condition, s.accuracy, s.precision, s.recall, s.f1_score,
                    s.fpr, s.fnr, s.avg_latency, s.enhancement_rate
                ])

    def generate_robustness_report(self, summaries: Dict[str, MetricsSummary]):
        report_path = os.path.join(self.results_dir, 'robustness_report.txt')
        
        overall = summaries.get('Overall')
        if not overall:
            return

        # Find weakest condition
        conditions = [s for k, s in summaries.items() if k != 'Overall']
        if conditions:
            weakest = min(conditions, key=lambda x: x.accuracy)
            strongest = max(conditions, key=lambda x: x.accuracy)
        else:
            weakest = overall
            strongest = overall

        with open(report_path, 'w') as f:
            f.write("Face Attendance System - Robustness Evaluation Report\n")
            f.write("====================================================\n\n")
            
            f.write(f"Overall Accuracy: {overall.accuracy * 100:.2f}%\n")
            f.write(f"Total Samples: {overall.total_samples}\n\n")
            
            f.write("1. Performance by Condition\n")
            f.write("---------------------------\n")
            for c in conditions:
                diff = c.accuracy - overall.accuracy
                f.write(f"- {c.condition}: {c.accuracy * 100:.2f}% (Deviation: {diff * 100:+.2f}%)\n")
            f.write("\n")
            
            f.write("2. Key Observations\n")
            f.write("-------------------\n")
            f.write(f"- Weakest Performing Condition: {weakest.condition} ({weakest.accuracy * 100:.2f}%)\n")
            f.write(f"- Strongest Performing Condition: {strongest.condition} ({strongest.accuracy * 100:.2f}%)\n")
            f.write(f"- Average Latency: {overall.avg_latency:.4f} seconds\n")
            f.write(f"- Enhancement Usage Rate: {overall.enhancement_rate * 100:.2f}%\n\n")
            
            f.write("3. Recommendations\n")
            f.write("------------------\n")
            if weakest.accuracy < 0.8:
                f.write(f"- Urgent improvement needed for '{weakest.condition}' scenarios.\n")
            if overall.enhancement_rate > 0.5:
                f.write("- High reliance on enhancement. Consider improving core detection model or camera quality.\n")
            f.write("- Collect more training data for failure cases.\n")

    def generate_bias_analysis(self, summaries: Dict[str, MetricsSummary]):
        report_path = os.path.join(self.results_dir, 'bias_analysis.txt')
        
        def safe_get_acc(cond):
            return summaries[cond].accuracy if cond in summaries else None

        comparisons = [
            ("Masked vs Non-Masked", "masked", "normal"), # Assuming 'normal' exists or we use Overall? Usually baseline is needed.
            ("Glasses vs Non-Glasses", "glasses", "normal"),
            ("Low-Light vs Normal", "low_light", "normal")
        ]
        
        # If 'normal' condition doesn't exist, we might compare against Overall or just skip
        # Let's check for "normal" or "baseline" folder, or imply it. 
        # The user didn't specify a "normal" folder in "Dataset Structure", but usually comparisons need a baseline.
        # I will check if 'normal' exists, otherwise compare to Overall.
        
        with open(report_path, 'w') as f:
            f.write("Bias Analysis Report\n")
            f.write("====================\n\n")
            
            for title, cond1, cond2 in comparisons:
                acc1 = safe_get_acc(cond1)
                # Try to find a baseline
                acc2 = safe_get_acc(cond2)
                if acc2 is None and 'Overall' in summaries:
                     acc2 = summaries['Overall'].accuracy # Use overall as proxy for average/normal
                     cond2 = "Overall Average"
                
                if acc1 is not None and acc2 is not None:
                    diff = acc1 - acc2
                    f.write(f"Comparison: {title}\n")
                    f.write(f"  {cond1}: {acc1 * 100:.2f}%\n")
                    f.write(f"  {cond2}: {acc2 * 100:.2f}%\n")
                    f.write(f"  Gap: {diff * 100:+.2f}%\n")
                    if abs(diff) > 0.1:
                        f.write("  Result: SIGNIFICANT BIAS DETECTED\n")
                    elif abs(diff) > 0.05:
                        f.write("  Result: MODERATE BIAS DETECTED\n")
                    else:
                        f.write("  Result: NEGLIGIBLE BIAS\n")
                    f.write("\n")
                else:
                    f.write(f"Comparison: {title} - SKIPPED (Missing data for {cond1} or {cond2})\n\n")
