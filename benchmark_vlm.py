"""
VLM Spatial Understanding Benchmark
=====================================
Benchmarks multiple Vision-Language Models on CV-Bench (2D/3D spatial tasks).
Evaluates: MCQ Accuracy, BLEU, ROUGE, METEOR, BERTScore.
Models are loaded sequentially with GPU memory cleared between each.

Usage:
    python benchmark_vlm.py --config config.yaml
"""

import os
import gc
import re
import json
import time
import argparse
import warnings
from pathlib import Path
from io import BytesIO

import yaml
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset

# Metrics
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import nltk

warnings.filterwarnings("ignore")

# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def clear_gpu_memory():
    """Aggressively clear GPU memory between model evaluations."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def extract_mcq_answer(text):
    """Extract MCQ answer (A/B/C/D) from model output."""
    text = text.strip()
    # Try to find pattern like (A), (B), etc.
    match = re.search(r'\(([A-D])\)', text)
    if match:
        return match.group(1)
    # Try standalone letter at start
    match = re.match(r'^([A-D])\b', text.strip())
    if match:
        return match.group(1)
    # Try letter followed by punctuation
    match = re.search(r'\b([A-D])[\.:\)\s]', text)
    if match:
        return match.group(1)
    # Last resort: check if any single letter A-D appears
    for letter in ['A', 'B', 'C', 'D']:
        if letter in text.upper().split():
            return letter
    return text.strip()[:1].upper() if text.strip() else ""


def format_prompt(question, choices):
    """Format the MCQ question with choices for the model."""
    if isinstance(choices, str):
        try:
            choices = json.loads(choices)
        except (json.JSONDecodeError, TypeError):
            pass

    if isinstance(choices, list):
        options = "\n".join([f"({chr(65+i)}) {c}" for i, c in enumerate(choices)])
    elif isinstance(choices, dict):
        options = "\n".join([f"({k}) {v}" for k, v in choices.items()])
    else:
        options = str(choices)

    prompt = f"{question}\n\nOptions:\n{options}\n\nAnswer with the letter of the correct option (A, B, C, or D)."
    return prompt


def extract_gt_answer(answer):
    """Extract ground truth answer letter from dataset answer field."""
    if isinstance(answer, str):
        match = re.search(r'\(([A-D])\)', answer)
        if match:
            return match.group(1)
        match = re.match(r'^([A-D])$', answer.strip())
        if match:
            return match.group(1)
        return answer.strip()[0].upper() if answer.strip() else ""
    return str(answer)


# ============================================================
# MODEL INFERENCE CLASSES (imported from vlm_bench.models)
# ============================================================

from vlm_bench.models import (
    InternVLInference,
    SmolVLMInference,
    SAILInference,
    LlamaVisionInference,
    QwenInference,
    get_inference_class,
)
from vlm_bench.utils import SPATIAL_SYSTEM_PROMPT


# ============================================================
# METRICS COMPUTATION
# ============================================================

class MetricsComputer:
    """Compute all evaluation metrics."""

    def __init__(self):
        # Download NLTK data
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab', quiet=True)
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet', quiet=True)

        self.rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.smoothing = SmoothingFunction().method1

    def compute_all(self, predictions, references, pred_answers, gt_answers):
        """Compute all metrics."""
        results = {}

        # MCQ Accuracy
        correct = sum(1 for p, g in zip(pred_answers, gt_answers) if p == g)
        results['mcq_accuracy'] = correct / len(gt_answers) if gt_answers else 0.0

        # Text metrics (on full generated text vs ground truth answer text)
        bleu_scores = []
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        meteor_scores = []

        for pred, ref in zip(predictions, references):
            pred_tokens = pred.lower().split()
            ref_tokens = ref.lower().split()

            # BLEU
            if pred_tokens and ref_tokens:
                bleu = sentence_bleu(
                    [ref_tokens], pred_tokens,
                    weights=(0.5, 0.5, 0, 0),
                    smoothing_function=self.smoothing
                )
            else:
                bleu = 0.0
            bleu_scores.append(bleu)

            # ROUGE
            rouge_result = self.rouge.score(ref, pred)
            rouge1_scores.append(rouge_result['rouge1'].fmeasure)
            rouge2_scores.append(rouge_result['rouge2'].fmeasure)
            rougeL_scores.append(rouge_result['rougeL'].fmeasure)

            # METEOR
            try:
                from nltk.translate.meteor_score import meteor_score
                m = meteor_score([ref_tokens], pred_tokens)
            except Exception:
                m = 0.0
            meteor_scores.append(m)

        results['bleu'] = np.mean(bleu_scores)
        results['rouge1'] = np.mean(rouge1_scores)
        results['rouge2'] = np.mean(rouge2_scores)
        results['rougeL'] = np.mean(rougeL_scores)
        results['meteor'] = np.mean(meteor_scores)

        # BERTScore
        try:
            from bert_score import score as bert_score_fn
            P, R, F1 = bert_score_fn(
                predictions, references,
                lang="en", verbose=False,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            results['bertscore_precision'] = P.mean().item()
            results['bertscore_recall'] = R.mean().item()
            results['bertscore_f1'] = F1.mean().item()
        except Exception as e:
            print(f"  [BERTScore Error]: {e}")
            results['bertscore_precision'] = 0.0
            results['bertscore_recall'] = 0.0
            results['bertscore_f1'] = 0.0

        return results

    def compute_per_task(self, df):
        """Compute metrics grouped by task type."""
        task_results = {}
        for task in df['task'].unique():
            task_df = df[df['task'] == task]
            correct = (task_df['pred_answer'] == task_df['gt_answer']).sum()
            task_results[task] = {
                'accuracy': correct / len(task_df) if len(task_df) > 0 else 0.0,
                'count': len(task_df)
            }
        return task_results


# ============================================================
# DATASET LOADING
# ============================================================

def load_cvbench_data(config):
    """Load CV-Bench dataset and filter by type."""
    print("\n" + "="*60)
    print("LOADING CV-BENCH DATASET")
    print("="*60)

    dataset = load_dataset(config['dataset']['name'], split=config['dataset']['split'])

    # Filter by type (2d/3d)
    type_filter = config['dataset'].get('type_filter', 'all')
    if type_filter != 'all':
        dataset = dataset.filter(lambda x: x['type'].lower() == type_filter.lower())
        print(f"  Filtered to type='{type_filter}': {len(dataset)} samples")

    # Take first N samples
    num_samples = config['dataset']['num_samples']
    if num_samples < len(dataset):
        dataset = dataset.select(range(num_samples))

    print(f"  Using {len(dataset)} samples for evaluation")

    # Print task distribution
    tasks = [s['task'] for s in dataset]
    task_counts = pd.Series(tasks).value_counts()
    print(f"\n  Task distribution:")
    for task, count in task_counts.items():
        print(f"    {task}: {count}")

    return dataset


# ============================================================
# MAIN BENCHMARKING LOOP
# ============================================================

def run_benchmark(config):
    """Run the full benchmarking pipeline."""
    # Create output directories
    results_dir = Path(config['output']['results_dir'])
    plots_dir = Path(config['output']['plots_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    dataset = load_cvbench_data(config)

    # Initialize metrics
    metrics_computer = MetricsComputer()

    # Store results for all models
    all_results = {}
    all_task_results = {}
    all_predictions_df = []

    # Evaluate each model sequentially
    for model_config in config['models']:
        model_name = model_config['name']
        model_path = model_config['path']
        model_type = model_config['type']

        print("\n" + "="*60)
        print(f"EVALUATING: {model_name}")
        print(f"  Path: {model_path}")
        print(f"  Type: {model_type}")
        print("="*60)

        # Load model
        print(f"\n  Loading model...")
        start_time = time.time()

        try:
            if model_type == "internvl":
                inference = InternVLInference(
                    model_path,
                    dtype=model_config.get('dtype', 'bfloat16'),
                    max_new_tokens=model_config.get('max_new_tokens', 128),
                )
            elif model_type == "smolvlm":
                inference = SmolVLMInference(
                    model_path,
                    dtype=model_config.get('dtype', 'bfloat16'),
                    max_new_tokens=model_config.get('max_new_tokens', 128),
                )
            elif model_type == "sail":
                inference = SAILInference(
                    model_path,
                    dtype=model_config.get('dtype', 'bfloat16'),
                    max_new_tokens=model_config.get('max_new_tokens', 128),
                )
            elif model_type == "llama":
                inference = LlamaVisionInference(
                    model_path,
                    load_in_4bit=model_config.get('load_in_4bit', True),
                    max_new_tokens=model_config.get('max_new_tokens', 128),
                )
            elif model_type == "qwen":
                inference = QwenInference(
                    model_path,
                    load_in_4bit=model_config.get('load_in_4bit', True),
                    max_new_tokens=model_config.get('max_new_tokens', 128),
                )
            else:
                print(f"  Unknown model type: {model_type}, skipping.")
                continue
        except Exception as e:
            print(f"  Failed to load model: {e}")
            continue

        load_time = time.time() - start_time
        print(f"  Model loaded in {load_time:.1f}s")

        # Run inference
        predictions = []
        pred_answers = []
        gt_answers = []
        references = []
        tasks = []
        inference_times = []

        print(f"\n  Running inference on {len(dataset)} samples...")
        for i, sample in enumerate(tqdm(dataset, desc=f"  {model_name}")):
            image = sample['image']
            if not isinstance(image, Image.Image):
                image = Image.open(BytesIO(image)).convert('RGB')
            else:
                image = image.convert('RGB')

            question = sample['question']
            choices = sample['choices']
            answer = sample['answer']
            task = sample['task']

            # Format prompt
            prompt = format_prompt(question, choices)

            # Generate
            t0 = time.time()
            response = inference.generate(image, prompt)
            inference_times.append(time.time() - t0)

            # Extract answers
            pred_letter = extract_mcq_answer(response)
            gt_letter = extract_gt_answer(answer)

            predictions.append(response)
            pred_answers.append(pred_letter)
            gt_answers.append(gt_letter)
            references.append(answer)
            tasks.append(task)

            # Progress logging every 25 samples
            if (i + 1) % 25 == 0:
                running_acc = sum(1 for p, g in zip(pred_answers, gt_answers) if p == g) / len(pred_answers)
                avg_time = np.mean(inference_times)
                print(f"    [{i+1}/{len(dataset)}] Running Acc: {running_acc:.3f} | Avg time: {avg_time:.2f}s/sample")

        # Compute metrics
        print(f"\n  Computing metrics...")
        results = metrics_computer.compute_all(predictions, references, pred_answers, gt_answers)
        results['avg_inference_time'] = np.mean(inference_times)
        results['total_inference_time'] = sum(inference_times)
        results['model_load_time'] = load_time

        all_results[model_name] = results

        # Per-task results
        pred_df = pd.DataFrame({
            'model': model_name,
            'task': tasks,
            'prediction': predictions,
            'pred_answer': pred_answers,
            'gt_answer': gt_answers,
            'reference': references,
        })
        all_predictions_df.append(pred_df)

        task_results = metrics_computer.compute_per_task(pred_df)
        all_task_results[model_name] = task_results

        # Print results
        print(f"\n  Results for {model_name}:")
        print(f"    MCQ Accuracy:      {results['mcq_accuracy']:.4f}")
        print(f"    BLEU:              {results['bleu']:.4f}")
        print(f"    ROUGE-1:           {results['rouge1']:.4f}")
        print(f"    ROUGE-2:           {results['rouge2']:.4f}")
        print(f"    ROUGE-L:           {results['rougeL']:.4f}")
        print(f"    METEOR:            {results['meteor']:.4f}")
        print(f"    BERTScore F1:      {results['bertscore_f1']:.4f}")
        print(f"    Avg Inference Time: {results['avg_inference_time']:.2f}s")

        print(f"\n  Per-task accuracy:")
        for task, task_res in task_results.items():
            print(f"    {task}: {task_res['accuracy']:.4f} ({task_res['count']} samples)")

        # Cleanup model from GPU
        print(f"\n  Clearing {model_name} from GPU memory...")
        inference.cleanup()
        clear_gpu_memory()
        print(f"  GPU memory cleared. Free: {torch.cuda.mem_get_info()[0]/1e9:.2f} GB")

    # ============================================================
    # SAVE RESULTS
    # ============================================================
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)

    # Save detailed predictions
    full_predictions_df = pd.concat(all_predictions_df, ignore_index=True)
    full_predictions_df.to_csv(results_dir / "detailed_predictions.csv", index=False)

    # Save summary table
    summary_df = pd.DataFrame(all_results).T
    summary_df.index.name = 'Model'
    summary_df.to_csv(results_dir / config['output']['results_file'])
    print(f"\n  Summary saved to: {results_dir / config['output']['results_file']}")

    # Save per-task results
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
    task_df.to_csv(results_dir / "per_task_results.csv", index=False)

    # Print final summary table
    print("\n" + "="*60)
    print("FINAL BENCHMARK RESULTS")
    print("="*60)
    print(f"\n{summary_df.to_string()}")

    print("\n\nPer-Task Accuracy:")
    task_pivot = task_df.pivot(index='Model', columns='Task', values='Accuracy')
    print(f"\n{task_pivot.to_string()}")

    # Save summary text
    with open(results_dir / config['output']['summary_file'], 'w') as f:
        f.write("VLM Spatial Understanding Benchmark Results\n")
        f.write("=" * 60 + "\n\n")
        f.write("Overall Metrics:\n")
        f.write(summary_df.to_string() + "\n\n")
        f.write("Per-Task Accuracy:\n")
        f.write(task_pivot.to_string() + "\n")

    # Generate plots
    generate_plots(all_results, all_task_results, plots_dir)

    return all_results, all_task_results


# ============================================================
# VISUALIZATION
# ============================================================

def generate_plots(all_results, all_task_results, plots_dir):
    """Generate comparison plots."""
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')

    models = list(all_results.keys())
    if not models:
        print("  No results to plot.")
        return

    # ---- Plot 1: Overall Metrics Comparison (Bar Chart) ----
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
        angles += angles[:1]  # Close the polygon

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

    # ---- Plot 5: MCQ Accuracy Bar with values ----
    fig, ax = plt.subplots(figsize=(8, 5))
    accuracies = [all_results[m]['mcq_accuracy'] for m in models]
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


# ============================================================
# ENTRY POINT
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="VLM Spatial Understanding Benchmark")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config YAML file')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    print("\n" + "="*60)
    print("VLM SPATIAL UNDERSTANDING BENCHMARK")
    print("="*60)
    print(f"  Config: {args.config}")
    print(f"  Models: {[m['name'] for m in config['models']]}")
    print(f"  Dataset: {config['dataset']['name']}")
    print(f"  Samples: {config['dataset']['num_samples']}")
    print(f"  Type filter: {config['dataset']['type_filter']}")
    print(f"  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"  VRAM: {torch.cuda.mem_get_info()[1]/1e9:.1f} GB" if torch.cuda.is_available() else "")

    # Run benchmark
    all_results, all_task_results = run_benchmark(config)

    print("\n" + "="*60)
    print("BENCHMARK COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
