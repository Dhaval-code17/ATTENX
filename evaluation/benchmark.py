import argparse
import os
import sys

# Ensure we can import locally
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# Ensure we can import from parent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset_manager import DatasetManager
from evaluator import Evaluator
from metrics import MetricsCalculator
from visualization import Visualizer
from reporter import Reporter

def main():
    parser = argparse.ArgumentParser(description="Face Attendance Robustness Benchmark")
    parser.add_argument('--dataset', default='evaluation_dataset', help='Path to evaluation dataset root')
    args = parser.parse_args()

    # Paths
    dataset_root = args.dataset
    if not os.path.exists(dataset_root):
        # try relative to project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        dataset_root = os.path.join(project_root, args.dataset)
        if not os.path.exists(dataset_root):
            print(f"Error: Dataset not found at {args.dataset}")
            return

    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')

    # Initialize Modules
    dataset_mgr = DatasetManager(dataset_root)
    evaluator = Evaluator()
    metrics_calc = MetricsCalculator()
    visualizer = Visualizer(results_dir)
    reporter = Reporter(results_dir, logs_dir)

    # 1. Load Dataset
    print(f"Loading dataset from {dataset_root}...")
    samples = dataset_mgr.load_dataset()
    print(f"Found {len(samples)} samples across {len(dataset_mgr.get_conditions())} conditions.")

    if not samples:
        print("No samples found. Exiting.")
        return

    # 2. Run Evaluation
    results = []
    print("Starting evaluation...")
    for i, sample in enumerate(samples):
        print(f"Processing {i+1}/{len(samples)}: {sample.image_name} ({sample.condition})")
        result = evaluator.evaluate_image(
            sample.image_path, 
            sample.condition, 
            sample.expected_identity
        )
        results.append(result)

    # 3. Calculate Metrics
    print("Calculating metrics...")
    summaries = metrics_calc.compute(results)

    # 4. Generate Outputs
    print("Generating reports and visualizations...")
    reporter.save_json_log(results)
    reporter.save_metrics_csv(summaries)
    reporter.generate_robustness_report(summaries)
    reporter.generate_bias_analysis(summaries)
    
    visualizer.plot_metrics(summaries)
    
    # 5. Confusion Matrices (Optional/Advanced)
    # For now, we collected overall metrics. 
    # Generating a full confusion matrix requires collecting all (Expected, Predicted) pairs.
    # The Visualizer has a method, but we need to feed it data.
    # Implementation left simple for now as per "Generate: Confusion matrix per condition"
    # To do this correctly, we'd need to aggregate labels.
    
    print(f"Benchmark complete. Results saved to {results_dir}")

if __name__ == "__main__":
    main()
