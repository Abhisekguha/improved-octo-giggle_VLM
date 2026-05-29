"""
Compare results from multiple model evaluations and generate plots.

Usage:
    python compare_models.py --config config.yaml
    python compare_models.py --results-dir ./results

Reads per-model result files (metrics_*.json, predictions_*.csv, per_task_*.csv)
produced by run_model.py and generates comparison tables and plots.
"""

import argparse
import json
import warnings
from pathlib import Path

import yaml
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def load_model_results(results_dir):
    """Load all model results from the results directory."""
    results_dir = Path(results_dir)

    all_metrics = {}
    all_predictions = []
    all_task_results = {}

    # Load metrics JSON files
    for metrics_file in sorted(results_dir.glob("metrics_*.json")):
        with open(metrics_file, 'r') as f:
            data = json.load(f)
        # Extract model name from filename: metrics_<name>.json
        model_name = metrics_file.stem.replace("metrics_", "")
        all_metrics[model_name] = data

    # Load prediction CSVs
    for pred_file in sorted(results_dir.glob("predictions_*.csv")):
        df = pd.read_csv(pred_file)
        all_predictions.append(df)

    # Load per-task CSVs
    for task_file in sorted(results_dir.glob("per_task_*.csv")):
        df = pd.read_csv(task_file)
        if not df.empty:
            model_name = df['Model'].iloc[0]
            task_dict = {}
            for _, row in df.iterrows():
                task_dict[row['Task']] = {
                    'accuracy': row['Accuracy'],
                    'count': int(row['Samples'])
                }
            all_task_results[model_name] = task_dict

    return all_metrics, all_predictions, all_task_results


def generate_plots(all_results, all_task_results, plots_dir):
    """Generate comparison plots."""
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')

    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    models = list(all_results.keys())
    if not models:
        print("  No results to plot.")
        return

    # ---- Plot 1: Overall Metrics Comparison ----
    fig, ax = plt.subplots(figsize=(12, 6))
    metrics_to_plot = ['mcq_accuracy', 'bleu', 'rouge1', 'rougeL', 'meteor', 'bertscore_f1']
    metric_labels = ['MCQ Acc', 'BLEU', 'ROUGE-1', 'ROUGE-L', 'METEOR', 'BERTScore F1']

    x = np.arange(len(metric_labels))
    width = 0.25
    offsets = np.linspace(-width, width, len(models))

    colors = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63']
    for i, model in enumerate(models):
        values = [all_results[model].get(m, 0) for m in metrics_to_plot]
        ax.bar(x + offsets[i], values, width, label=model, color=colors[i % len(colors)])

    ax.set_xlabel('Metric')
    ax.set_ylabel('Score')
    ax.set_title('VLM Benchmark: Overall Metrics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "overall_metrics_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {plots_dir / 'overall_metrics_comparison.png'}")

    # ---- Plot 2: Per-Task Accuracy Comparison ----
    all_tasks = set()
    for model_tasks in all_task_results.values():
        all_tasks.update(model_tasks.keys())
    all_tasks = sorted(all_tasks)

    if all_tasks:
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(all_tasks))
        offsets = np.linspace(-width, width, len(models))

        for i, model in enumerate(models):
            values = [all_task_results[model].get(t, {}).get('accuracy', 0) for t in all_tasks]
            ax.bar(x + offsets[i], values, width, label=model, color=colors[i % len(colors)])

        ax.set_xlabel('Task Type')
        ax.set_ylabel('Accuracy')
        ax.set_title('VLM Benchmark: Per-Task Accuracy Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(all_tasks, rotation=30, ha='right')
        ax.legend(loc='upper right')
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / "per_task_accuracy_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {plots_dir / 'per_task_accuracy_comparison.png'}")

    # ---- Plot 3: Radar Chart ----
    if all_tasks and len(all_tasks) >= 3:
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        angles = np.linspace(0, 2 * np.pi, len(all_tasks), endpoint=False).tolist()
        angles += angles[:1]

        for i, model in enumerate(models):
            values = [all_task_results[model].get(t, {}).get('accuracy', 0) for t in all_tasks]
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[i % len(colors)])
            ax.fill(angles, values, alpha=0.1, color=colors[i % len(colors)])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(all_tasks, size=9)
        ax.set_ylim(0, 1.0)
        ax.set_title('Spatial Task Performance Radar', size=14, pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        plt.tight_layout()
        plt.savefig(plots_dir / "radar_task_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {plots_dir / 'radar_task_comparison.png'}")

    # ---- Plot 4: Inference Speed Comparison ----
    fig, ax = plt.subplots(figsize=(8, 5))
    times = [all_results[m].get('avg_inference_time', 0) for m in models]
    bars = ax.barh(models, times, color=colors[:len(models)])
    ax.set_xlabel('Avg Inference Time (seconds/sample)')
    ax.set_title('Inference Speed Comparison')
    ax.grid(axis='x', alpha=0.3)
    for bar, t in zip(bars, times):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                f'{t:.2f}s', va='center')
    plt.tight_layout()
    plt.savefig(plots_dir / "inference_speed_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {plots_dir / 'inference_speed_comparison.png'}")

    # ---- Plot 5: MCQ Accuracy Bar ----
    fig, ax = plt.subplots(figsize=(8, 5))
    accuracies = [all_results[m].get('mcq_accuracy', 0) for m in models]
    bars = ax.bar(models, accuracies, color=colors[:len(models)], edgecolor='black', linewidth=0.5)
    ax.set_ylabel('MCQ Accuracy')
    ax.set_title('MCQ Accuracy Comparison — Spatial Understanding')
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{acc:.3f}', ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(plots_dir / "mcq_accuracy_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {plots_dir / 'mcq_accuracy_comparison.png'}")


def main():
    parser = argparse.ArgumentParser(description="Compare VLM benchmark results across models")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config YAML (for output paths)')
    parser.add_argument('--results-dir', type=str, default=None,
                        help='Override results directory (default: from config)')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    results_dir = Path(args.results_dir) if args.results_dir else Path(config['output']['results_dir'])
    plots_dir = Path(config['output']['plots_dir'])

    print("\n" + "="*60)
    print("VLM BENCHMARK — MODEL COMPARISON")
    print("="*60)
    print(f"  Results dir: {results_dir}")
    print(f"  Plots dir:   {plots_dir}")

    # Load all model results
    all_metrics, all_predictions, all_task_results = load_model_results(results_dir)

    if not all_metrics:
        print("\n  ERROR: No model results found in the results directory.")
        print("  Run individual model evaluations first:")
        print("    python run_model.py --config config.yaml --model <model_name>")
        return

    print(f"\n  Found results for {len(all_metrics)} models: {list(all_metrics.keys())}")

    # Build summary table
    summary_df = pd.DataFrame(all_metrics).T
    summary_df.index.name = 'Model'
    summary_df.to_csv(results_dir / config['output']['results_file'])
    print(f"\n  Summary saved to: {results_dir / config['output']['results_file']}")

    # Combine all predictions
    if all_predictions:
        full_predictions_df = pd.concat(all_predictions, ignore_index=True)
        full_predictions_df.to_csv(results_dir / "detailed_predictions.csv", index=False)

    # Per-task pivot table
    task_rows = []
    for model_name, tasks in all_task_results.items():
        for task, res in tasks.items():
            task_rows.append({
                'Model': model_name,
                'Task': task,
                'Accuracy': res['accuracy'],
                'Samples': res['count']
            })
    task_df = pd.DataFrame(task_rows)
    if not task_df.empty:
        task_df.to_csv(results_dir / "per_task_results.csv", index=False)

    # Print final summary
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    print(f"\n{summary_df.to_string()}")

    if not task_df.empty:
        print("\n\nPer-Task Accuracy:")
        task_pivot = task_df.pivot(index='Model', columns='Task', values='Accuracy')
        print(f"\n{task_pivot.to_string()}")

    # Save summary text
    with open(results_dir / config['output']['summary_file'], 'w') as f:
        f.write("VLM Spatial Understanding Benchmark Results\n")
        f.write("=" * 60 + "\n\n")
        f.write("Overall Metrics:\n")
        f.write(summary_df.to_string() + "\n\n")
        if not task_df.empty:
            f.write("Per-Task Accuracy:\n")
            task_pivot = task_df.pivot(index='Model', columns='Task', values='Accuracy')
            f.write(task_pivot.to_string() + "\n")

    # Generate plots
    print("\n  Generating comparison plots...")
    generate_plots(all_metrics, all_task_results, plots_dir)

    print("\n" + "="*60)
    print("COMPARISON COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
