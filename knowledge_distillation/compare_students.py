"""
Compare multiple distilled student models and generate comparison plots.

Reads evaluation results from evaluate_student.py outputs and creates:
  - Side-by-side metrics comparison (bar charts)
  - Per-task accuracy comparison
  - Radar chart comparison
  - Inference speed comparison
  - Summary CSV and text reports

Usage:
    python knowledge_distillation/compare_students.py \
        --result_dirs "eval_results/InternVL3_5-1B-Instruct_LoRA" \
                      "eval_results/SmolVLM-Instruct-KD" \
        --model_names "InternVL-1B-KD" "SmolVLM-2B-KD" \
        --output_dir "knowledge_distillation/comparison_results"
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_model_results(result_dir):
    """Load evaluation results from a model's result directory."""
    result_dir = Path(result_dir)
    
    # Find metrics JSON file
    metrics_files = list(result_dir.glob("metrics_*.json"))
    if not metrics_files:
        raise FileNotFoundError(f"No metrics_*.json found in {result_dir}")
    
    with open(metrics_files[0], 'r') as f:
        metrics = json.load(f)
    
    # Load per-task results if available
    per_task_file = result_dir / "per_task_results.csv"
    if per_task_file.exists():
        per_task_df = pd.read_csv(per_task_file)
    else:
        per_task_df = None
    
    # Load detailed predictions if available
    pred_file = result_dir / "detailed_predictions.csv"
    if pred_file.exists():
        predictions_df = pd.read_csv(pred_file)
    else:
        predictions_df = None
    
    return {
        'metrics': metrics,
        'per_task_df': per_task_df,
        'predictions_df': predictions_df,
        'result_dir': result_dir
    }


def generate_overall_metrics_comparison(all_results, model_names, output_dir):
    """Generate bar chart comparing overall metrics across models."""
    metric_keys = ['mcq_accuracy', 'bleu', 'rouge1', 'rougeL', 'meteor', 'bertscore_f1']
    metric_labels = ['MCQ Acc', 'BLEU', 'ROUGE-1', 'ROUGE-L', 'METEOR', 'BERTScore F1']
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    x = np.arange(len(metric_labels))
    width = 0.8 / len(model_names)
    offsets = np.linspace(-0.4, 0.4, len(model_names))
    
    colors = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63', '#9C27B0', '#00BCD4']
    
    for i, (model_name, results) in enumerate(zip(model_names, all_results)):
        metrics = results['metrics']
        values = [metrics.get(k, 0.0) for k in metric_keys]
        bars = ax.bar(x + offsets[i], values, width, label=model_name, 
                     color=colors[i % len(colors)], edgecolor='black', linewidth=0.5)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            if val > 0.05:  # Only show if significant
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=8, rotation=0)
    
    ax.set_xlabel('Metric', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Knowledge Distillation Student Models — Metrics Comparison', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    output_path = output_dir / "overall_metrics_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def generate_per_task_comparison(all_results, model_names, output_dir):
    """Generate grouped bar chart comparing per-task accuracy."""
    # Collect all unique tasks
    all_tasks = set()
    for results in all_results:
        if results['per_task_df'] is not None:
            all_tasks.update(results['per_task_df']['Task'].unique())
    
    if not all_tasks:
        print("  Skipping per-task comparison (no task data available)")
        return
    
    all_tasks = sorted(all_tasks)
    
    fig, ax = plt.subplots(figsize=(12, max(6, len(all_tasks) * 0.5)))
    
    y = np.arange(len(all_tasks))
    width = 0.8 / len(model_names)
    offsets = np.linspace(-0.4, 0.4, len(model_names))
    
    colors = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63', '#9C27B0', '#00BCD4']
    
    for i, (model_name, results) in enumerate(zip(model_names, all_results)):
        if results['per_task_df'] is None:
            continue
        
        task_accs = []
        for task in all_tasks:
            task_row = results['per_task_df'][results['per_task_df']['Task'] == task]
            if not task_row.empty:
                task_accs.append(task_row['Accuracy'].values[0])
            else:
                task_accs.append(0.0)
        
        bars = ax.barh(y + offsets[i], task_accs, width, label=model_name,
                      color=colors[i % len(colors)], edgecolor='black', linewidth=0.5)
        
        # Add value labels
        for bar, acc in zip(bars, task_accs):
            if acc > 0.05:
                ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                       f'{acc:.3f}', ha='left', va='center', fontsize=8)
    
    ax.set_yticks(y)
    ax.set_yticklabels(all_tasks, fontsize=10)
    ax.set_xlabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Per-Task Accuracy Comparison', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlim(0, 1.0)
    ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    output_path = output_dir / "per_task_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def generate_radar_comparison(all_results, model_names, output_dir):
    """Generate radar chart comparing metrics across models."""
    metric_keys = ['mcq_accuracy', 'bleu', 'rouge1', 'rougeL', 'meteor', 'bertscore_f1']
    metric_labels = ['MCQ Acc', 'BLEU', 'ROUGE-1', 'ROUGE-L', 'METEOR', 'BERTScore']
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    angles = np.linspace(0, 2 * np.pi, len(metric_labels), endpoint=False).tolist()
    angles += [angles[0]]  # Close the plot
    
    colors = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63', '#9C27B0', '#00BCD4']
    
    for i, (model_name, results) in enumerate(zip(model_names, all_results)):
        metrics = results['metrics']
        values = [metrics.get(k, 0.0) for k in metric_keys]
        values += [values[0]]  # Close the plot
        
        color = colors[i % len(colors)]
        ax.fill(angles, values, alpha=0.15, color=color, label=model_name)
        ax.plot(angles, values, 'o-', color=color, linewidth=2, markersize=6)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_title('Metrics Radar Comparison — KD Students', 
                y=1.08, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    plt.tight_layout()
    
    output_path = output_dir / "metrics_radar_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def generate_inference_speed_comparison(all_results, model_names, output_dir):
    """Generate bar chart comparing inference speed."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    avg_times = []
    total_times = []
    
    for results in all_results:
        metrics = results['metrics']
        avg_times.append(metrics.get('avg_inference_time', 0.0))
        total_times.append(metrics.get('total_inference_time', 0.0))
    
    x = np.arange(len(model_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, avg_times, width, label='Avg Time/Sample (s)',
                   color='#FF9800', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, [t/60 for t in total_times], width, 
                   label='Total Time (min)', color='#4CAF50', edgecolor='black', linewidth=0.5)
    
    # Add value labels
    for bar, val in zip(bars1, avg_times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
               f'{val:.2f}s', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    for bar, val in zip(bars2, total_times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 60 + 0.5,
               f'{val/60:.1f}m', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_ylabel('Time', fontsize=12, fontweight='bold')
    ax.set_title('Inference Speed Comparison', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=11)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    output_path = output_dir / "inference_speed_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def generate_mcq_accuracy_focused(all_results, model_names, output_dir):
    """Generate focused bar chart for MCQ accuracy (primary metric)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    accuracies = [results['metrics'].get('mcq_accuracy', 0.0) for results in all_results]
    
    colors = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63', '#9C27B0', '#00BCD4']
    bars = ax.bar(model_names, accuracies, 
                  color=[colors[i % len(colors)] for i in range(len(model_names))],
                  edgecolor='black', linewidth=1.5, width=0.6)
    
    # Add value labels and percentage
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.01,
               f'{acc:.4f}\n({acc*100:.2f}%)', 
               ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('MCQ Accuracy', fontsize=13, fontweight='bold')
    ax.set_title('MCQ Accuracy Comparison — Spatial Understanding (Primary Metric)', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0, min(1.0, max(accuracies) * 1.2))
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    
    output_path = output_dir / "mcq_accuracy_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def generate_summary_tables(all_results, model_names, output_dir):
    """Generate CSV and text summary of comparison."""
    # Build comparison dataframe
    rows = []
    for model_name, results in zip(model_names, all_results):
        metrics = results['metrics']
        rows.append({
            'Model': model_name,
            'MCQ_Accuracy': metrics.get('mcq_accuracy', 0.0),
            'BLEU': metrics.get('bleu', 0.0),
            'ROUGE-1': metrics.get('rouge1', 0.0),
            'ROUGE-2': metrics.get('rouge2', 0.0),
            'ROUGE-L': metrics.get('rougeL', 0.0),
            'METEOR': metrics.get('meteor', 0.0),
            'BERTScore_F1': metrics.get('bertscore_f1', 0.0),
            'Avg_Inference_Time': metrics.get('avg_inference_time', 0.0),
            'Total_Inference_Time': metrics.get('total_inference_time', 0.0),
            'Num_Samples': metrics.get('num_samples', 0),
        })
    
    df = pd.DataFrame(rows)
    
    # Save CSV
    csv_path = output_dir / "comparison_summary.csv"
    df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"  Saved: {csv_path}")
    
    # Save text summary
    txt_path = output_dir / "comparison_summary.txt"
    with open(txt_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("KNOWLEDGE DISTILLATION STUDENT MODELS COMPARISON\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Models Compared: {len(model_names)}\n")
        for i, name in enumerate(model_names, 1):
            f.write(f"  {i}. {name}\n")
        f.write("\n")
        
        f.write("-" * 80 + "\n")
        f.write("OVERALL METRICS\n")
        f.write("-" * 80 + "\n")
        f.write(df.to_string(index=False))
        f.write("\n\n")
        
        # Best model per metric
        f.write("-" * 80 + "\n")
        f.write("BEST MODEL PER METRIC\n")
        f.write("-" * 80 + "\n")
        metric_cols = ['MCQ_Accuracy', 'BLEU', 'ROUGE-1', 'ROUGE-L', 'METEOR', 'BERTScore_F1']
        for col in metric_cols:
            best_idx = df[col].idxmax()
            best_model = df.loc[best_idx, 'Model']
            best_val = df.loc[best_idx, col]
            f.write(f"  {col:20s}: {best_model:30s} ({best_val:.4f})\n")
        
        f.write("\n")
        f.write("-" * 80 + "\n")
        f.write("INFERENCE SPEED\n")
        f.write("-" * 80 + "\n")
        fastest_idx = df['Avg_Inference_Time'].idxmin()
        fastest_model = df.loc[fastest_idx, 'Model']
        fastest_time = df.loc[fastest_idx, 'Avg_Inference_Time']
        f.write(f"  Fastest: {fastest_model} ({fastest_time:.2f}s/sample)\n")
        
        slowest_idx = df['Avg_Inference_Time'].idxmax()
        slowest_model = df.loc[slowest_idx, 'Model']
        slowest_time = df.loc[slowest_idx, 'Avg_Inference_Time']
        f.write(f"  Slowest: {slowest_model} ({slowest_time:.2f}s/sample)\n")
        
        # Per-task comparison if available
        f.write("\n")
        f.write("-" * 80 + "\n")
        f.write("PER-TASK ACCURACY\n")
        f.write("-" * 80 + "\n")
        
        # Collect all tasks
        all_tasks = set()
        for results in all_results:
            if results['per_task_df'] is not None:
                all_tasks.update(results['per_task_df']['Task'].unique())
        
        if all_tasks:
            all_tasks = sorted(all_tasks)
            for task in all_tasks:
                f.write(f"\n  {task}:\n")
                for model_name, results in zip(model_names, all_results):
                    if results['per_task_df'] is not None:
                        task_row = results['per_task_df'][results['per_task_df']['Task'] == task]
                        if not task_row.empty:
                            acc = task_row['Accuracy'].values[0]
                            count = task_row['Count'].values[0]
                            f.write(f"    {model_name:30s}: {acc:.4f} ({int(count)} samples)\n")
        else:
            f.write("  (No per-task data available)\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    print(f"  Saved: {txt_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare multiple KD student model evaluations and generate plots"
    )
    parser.add_argument(
        '--result_dirs', nargs='+', required=True,
        help='Paths to evaluation result directories (from evaluate_student.py)'
    )
    parser.add_argument(
        '--model_names', nargs='+', required=True,
        help='Display names for each model (must match order of result_dirs)'
    )
    parser.add_argument(
        '--output_dir', type=str, 
        default='knowledge_distillation/comparison_results',
        help='Directory to save comparison plots and summary'
    )
    args = parser.parse_args()
    
    if len(args.result_dirs) != len(args.model_names):
        raise ValueError(
            f"Number of result_dirs ({len(args.result_dirs)}) must match "
            f"number of model_names ({len(args.model_names)})"
        )
    
    print("\n" + "=" * 80)
    print("KNOWLEDGE DISTILLATION STUDENT COMPARISON")
    print("=" * 80)
    print(f"  Models: {len(args.model_names)}")
    for name, dir_path in zip(args.model_names, args.result_dirs):
        print(f"    - {name:30s} → {dir_path}")
    print(f"  Output: {args.output_dir}")
    print("=" * 80)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all results
    print("\n  Loading evaluation results...")
    all_results = []
    for result_dir in args.result_dirs:
        try:
            results = load_model_results(result_dir)
            all_results.append(results)
            print(f"    ✓ Loaded: {result_dir}")
        except Exception as e:
            print(f"    ✗ Failed to load {result_dir}: {e}")
            raise
    
    # Generate comparison plots
    print("\n  Generating comparison plots...")
    
    generate_mcq_accuracy_focused(all_results, args.model_names, output_dir)
    generate_overall_metrics_comparison(all_results, args.model_names, output_dir)
    generate_per_task_comparison(all_results, args.model_names, output_dir)
    generate_radar_comparison(all_results, args.model_names, output_dir)
    generate_inference_speed_comparison(all_results, args.model_names, output_dir)
    
    # Generate summary tables
    print("\n  Generating summary tables...")
    generate_summary_tables(all_results, args.model_names, output_dir)
    
    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)
    print(f"\n  All results saved to: {output_dir}")
    print(f"  - comparison_summary.csv (metrics table)")
    print(f"  - comparison_summary.txt (detailed report)")
    print(f"  - *.png (5 comparison plots)")
    print()


if __name__ == "__main__":
    main()
