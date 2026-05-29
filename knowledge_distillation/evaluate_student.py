"""
Step 3: Evaluate the distilled student model on CV-Bench test set.

Full benchmarking with:
  - MCQ Accuracy (overall + per-task)
  - BLEU, ROUGE-1/2/L, METEOR, BERTScore
  - Comparison plots saved to output directory
  - CSV/JSON results export

Supports InternVL (chat-based) and SmolVLM (generate-based) student models.
"""

import argparse
import gc
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm

# Metrics
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

warnings.filterwarnings("ignore")

# Support both direct execution and module execution
try:
    from knowledge_distillation.config import StudentConfig, DataConfig, PathConfig
    from knowledge_distillation.utils import (
        extract_mcq_answer,
        format_question,
        load_cvbench,
    )
except ModuleNotFoundError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from knowledge_distillation.config import StudentConfig, DataConfig, PathConfig
    from knowledge_distillation.utils import (
        extract_mcq_answer,
        format_question,
        load_cvbench,
    )


# ------------------------------------------------------------------
# GPU Memory Management
# ------------------------------------------------------------------

def clear_gpu_memory():
    """Aggressively clear GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


# ------------------------------------------------------------------
# Metrics Computation (aligned with benchmark_vlm.py / vlm_bench.metrics)
# ------------------------------------------------------------------

class MetricsComputer:
    """Compute all evaluation metrics: MCQ Accuracy, BLEU, ROUGE, METEOR, BERTScore."""

    def __init__(self):
        import nltk
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

        results['bleu'] = np.mean(bleu_scores) if bleu_scores else 0.0
        results['rouge1'] = np.mean(rouge1_scores) if rouge1_scores else 0.0
        results['rouge2'] = np.mean(rouge2_scores) if rouge2_scores else 0.0
        results['rougeL'] = np.mean(rougeL_scores) if rougeL_scores else 0.0
        results['meteor'] = np.mean(meteor_scores) if meteor_scores else 0.0

        # BERTScore
        try:
            from bert_score import score as bert_score_fn
            P, R, F1 = bert_score_fn(
                predictions, references,
                model_type="microsoft/deberta-xlarge-mnli",
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


# ------------------------------------------------------------------
# Image preprocessing (from InternVL)
# ------------------------------------------------------------------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1)
        for i in range(1, n + 1) for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = orig_width * orig_height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio

    target_width = image_size * best_ratio[0]
    target_height = image_size * best_ratio[1]
    blocks = best_ratio[0] * best_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)

    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)

    return processed_images


def load_image(image_input, input_size=448, max_num=12):
    """Load and preprocess image for InternVL."""
    if isinstance(image_input, str):
        image = Image.open(image_input).convert('RGB')
    elif isinstance(image_input, Image.Image):
        image = image_input.convert('RGB')
    else:
        raise ValueError("image_input must be a file path or PIL Image")

    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


# ------------------------------------------------------------------
# Model loading
# ------------------------------------------------------------------

def load_student_internvl(model_path: str, adapter_path: str = None):
    """Load InternVL-based student model (base + optional LoRA adapter)."""
    from transformers import AutoModel, AutoTokenizer

    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )

    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    return model, tokenizer, None


def load_student_smolvlm(model_path: str, adapter_path: str = None):
    """Load SmolVLM-based student model (base + optional LoRA adapter)."""
    from transformers import AutoProcessor, AutoModelForImageTextToText

    processor = AutoProcessor.from_pretrained(model_path)
    torch_dtype = torch.bfloat16

    try:
        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            _attn_implementation="flash_attention_2",
        ).to("cuda")
    except Exception:
        model = AutoModelForImageTextToText.from_pretrained(
            model_path, torch_dtype=torch_dtype
        ).to("cuda")

    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, None, processor


# ------------------------------------------------------------------
# Inference helpers
# ------------------------------------------------------------------

def generate_internvl(model, tokenizer, image, query, max_new_tokens=128):
    """Generate response using InternVL chat interface."""
    pixel_values = load_image(image, max_num=12).to(torch.bfloat16)
    if torch.cuda.is_available():
        pixel_values = pixel_values.cuda()
    generation_config = {"max_new_tokens": max_new_tokens, "do_sample": False}
    response = model.chat(tokenizer, pixel_values, query, generation_config)
    return response


def generate_smolvlm(model, processor, image, query, max_new_tokens=128):
    """Generate response using SmolVLM processor-based pipeline."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": query},
            ]
        },
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=torch.bfloat16)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs, do_sample=False, max_new_tokens=max_new_tokens
        )

    input_len = inputs["input_ids"].shape[1]
    response = processor.batch_decode(
        output_ids[:, input_len:], skip_special_tokens=True
    )[0]
    return response.strip()


# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------

def evaluate(model, tokenizer, processor, dataset, model_type="internvl", max_new_tokens=128):
    """Evaluate model on CV-Bench and return full metrics + predictions."""
    predictions_text = []
    pred_answers = []
    gt_answers = []
    references = []
    tasks = []
    inference_times = []

    print(f"\n  Running inference on {len(dataset)} samples...")
    for idx in tqdm(range(len(dataset)), desc="Evaluating"):
        sample = dataset[idx]
        image = sample["image"]
        question = sample["question"]
        options = sample.get("choices", sample.get("options", []))
        answer = sample["answer"]
        task = sample.get("task", sample.get("type", "unknown"))

        formatted_q = format_question(question, options)
        query = f"<image>\n{formatted_q}"

        start_time = time.time()
        try:
            if model_type == "internvl":
                response = generate_internvl(model, tokenizer, image, query, max_new_tokens)
            elif model_type == "smolvlm":
                response = generate_smolvlm(model, processor, image, query, max_new_tokens)
            else:
                # Fallback: try chat interface, then generate
                if hasattr(model, "chat") and tokenizer is not None:
                    response = generate_internvl(model, tokenizer, image, query, max_new_tokens)
                elif processor is not None:
                    response = generate_smolvlm(model, processor, image, query, max_new_tokens)
                else:
                    response = ""
        except Exception as e:
            response = ""
            print(f"  [Error idx={idx}]: {e}")

        elapsed = time.time() - start_time
        inference_times.append(elapsed)

        pred_letter = extract_mcq_answer(response)
        gt_letter = extract_mcq_answer(str(answer))

        predictions_text.append(response)
        pred_answers.append(pred_letter)
        gt_answers.append(gt_letter)
        references.append(str(answer))
        tasks.append(task)

    return {
        "predictions_text": predictions_text,
        "pred_answers": pred_answers,
        "gt_answers": gt_answers,
        "references": references,
        "tasks": tasks,
        "inference_times": inference_times,
    }


# ------------------------------------------------------------------
# Plot Generation (aligned with benchmark_vlm.py)
# ------------------------------------------------------------------

def generate_plots(model_name, metrics, task_results, plots_dir):
    """Generate benchmark plots for a single model evaluation."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(plots_dir, exist_ok=True)

    # ---- Plot 1: Overall Metrics Bar Chart ----
    fig, ax = plt.subplots(figsize=(10, 6))
    metric_keys = ['mcq_accuracy', 'bleu', 'rouge1', 'rougeL', 'meteor', 'bertscore_f1']
    metric_labels = ['MCQ Acc', 'BLEU', 'ROUGE-1', 'ROUGE-L', 'METEOR', 'BERTScore F1']
    values = [metrics.get(k, 0.0) for k in metric_keys]

    colors = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63', '#9C27B0', '#00BCD4']
    bars = ax.bar(metric_labels, values, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Score')
    ax.set_title(f'Benchmark Metrics — {model_name}')
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "overall_metrics.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {plots_dir}/overall_metrics.png")

    # ---- Plot 2: Per-Task Accuracy ----
    if task_results:
        fig, ax = plt.subplots(figsize=(10, 6))
        task_names = list(task_results.keys())
        task_accs = [task_results[t]['accuracy'] for t in task_names]

        bars = ax.barh(task_names, task_accs, color='#2196F3', edgecolor='black', linewidth=0.5)
        ax.set_xlabel('Accuracy')
        ax.set_title(f'Per-Task Accuracy — {model_name}')
        ax.set_xlim(0, 1.0)
        ax.grid(axis='x', alpha=0.3)
        for bar, acc in zip(bars, task_accs):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                    f'{acc:.3f}', ha='left', va='center', fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "per_task_accuracy.png"), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {plots_dir}/per_task_accuracy.png")

    # ---- Plot 3: Inference Speed Distribution ----
    if metrics.get('avg_inference_time', 0) > 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        speed_data = [
            ('Avg Time/Sample', metrics['avg_inference_time']),
            ('Total Time', metrics['total_inference_time']),
        ]
        labels_s = [s[0] for s in speed_data]
        vals_s = [s[1] for s in speed_data]
        ax.bar(labels_s, vals_s, color=['#FF9800', '#4CAF50'], edgecolor='black', linewidth=0.5)
        ax.set_ylabel('Time (seconds)')
        ax.set_title(f'Inference Timing — {model_name}')
        ax.grid(axis='y', alpha=0.3)
        for i, v in enumerate(vals_s):
            ax.text(i, v + 0.5, f'{v:.2f}s', ha='center', fontsize=10, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "inference_timing.png"), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {plots_dir}/inference_timing.png")

    # ---- Plot 4: Radar Chart of Metrics ----
    if len(metric_keys) >= 3:
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        angles = np.linspace(0, 2 * np.pi, len(metric_labels), endpoint=False).tolist()
        values_radar = values + [values[0]]
        angles += [angles[0]]
        ax.fill(angles, values_radar, alpha=0.25, color='#2196F3')
        ax.plot(angles, values_radar, 'o-', color='#2196F3', linewidth=2)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels, fontsize=10)
        ax.set_ylim(0, 1.0)
        ax.set_title(f'Metrics Radar — {model_name}', y=1.08, fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "metrics_radar.png"), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {plots_dir}/metrics_radar.png")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate distilled student model (full benchmark)")
    parser.add_argument("--student_path", type=str, default="OpenGVLab/InternVL3_5-1B-Instruct")
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--model_type", type=str, choices=["internvl", "smolvlm", "auto"],
                        default="auto", help="Model architecture type")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Display name for the model (used in plots/results)")
    parser.add_argument("--dataset", type=str, default="nyu-visionx/CV-Bench")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--filter_type", type=str, default="2D",
                        help="Filter dataset by type (2D, 3D, or all)")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--output_dir", type=str,
                        default="knowledge_distillation/eval_results")
    args = parser.parse_args()

    # Determine model name for display
    model_name = args.model_name or os.path.basename(args.student_path)
    if args.adapter_path:
        model_name += "_LoRA"

    # Auto-detect model type
    if args.model_type == "auto":
        path_lower = args.student_path.lower()
        if "smolvlm" in path_lower or "smol" in path_lower:
            model_type = "smolvlm"
        else:
            model_type = "internvl"
    else:
        model_type = args.model_type

    print("\n" + "=" * 60)
    print("  STUDENT MODEL BENCHMARK EVALUATION")
    print("=" * 60)
    print(f"  Model      : {args.student_path}")
    print(f"  Adapter    : {args.adapter_path or 'None (baseline)'}")
    print(f"  Model Type : {model_type}")
    print(f"  Display    : {model_name}")
    print(f"  Dataset    : {args.dataset} [{args.split}]")
    print(f"  Filter     : {args.filter_type}")
    print(f"  Max Tokens : {args.max_new_tokens}")
    print(f"  GPU        : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"  VRAM       : {torch.cuda.mem_get_info()[1]/1e9:.1f} GB")
    print("=" * 60)

    # Create output directories
    results_dir = os.path.join(args.output_dir, model_name.replace(" ", "_"))
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # Load dataset
    filter_type = args.filter_type if args.filter_type.lower() != "all" else None
    dataset = load_cvbench(
        args.dataset, args.split, filter_type=filter_type, max_samples=args.max_samples
    )
    print(f"\n  Evaluation samples: {len(dataset)}")

    # Print task distribution
    task_list = [dataset[i].get("task", dataset[i].get("type", "unknown")) for i in range(len(dataset))]
    task_counts = pd.Series(task_list).value_counts()
    print(f"  Task distribution:")
    for task, count in task_counts.items():
        print(f"    {task}: {count}")

    # Load model
    print(f"\n  Loading model...")
    load_start = time.time()
    if model_type == "smolvlm":
        model, tokenizer, processor = load_student_smolvlm(args.student_path, args.adapter_path)
    else:
        model, tokenizer, processor = load_student_internvl(args.student_path, args.adapter_path)
    load_time = time.time() - load_start
    print(f"  Model loaded in {load_time:.1f}s")

    # Run evaluation
    eval_output = evaluate(
        model, tokenizer, processor, dataset,
        model_type=model_type, max_new_tokens=args.max_new_tokens
    )

    # Compute metrics
    print(f"\n  Computing metrics...")
    metrics_computer = MetricsComputer()
    metrics = metrics_computer.compute_all(
        eval_output["predictions_text"],
        eval_output["references"],
        eval_output["pred_answers"],
        eval_output["gt_answers"],
    )
    metrics['avg_inference_time'] = np.mean(eval_output["inference_times"])
    metrics['total_inference_time'] = sum(eval_output["inference_times"])
    metrics['model_load_time'] = load_time
    metrics['num_samples'] = len(dataset)

    # Per-task results
    pred_df = pd.DataFrame({
        'model': model_name,
        'task': eval_output["tasks"],
        'prediction': eval_output["predictions_text"],
        'pred_answer': eval_output["pred_answers"],
        'gt_answer': eval_output["gt_answers"],
        'reference': eval_output["references"],
    })
    task_results = metrics_computer.compute_per_task(pred_df)

    # Print results
    print("\n" + "=" * 60)
    print("  BENCHMARK RESULTS")
    print("=" * 60)
    print(f"  Model: {model_name}")
    print(f"  {'─' * 40}")
    print(f"  MCQ Accuracy:       {metrics['mcq_accuracy']:.4f}")
    print(f"  BLEU:               {metrics['bleu']:.4f}")
    print(f"  ROUGE-1:            {metrics['rouge1']:.4f}")
    print(f"  ROUGE-2:            {metrics['rouge2']:.4f}")
    print(f"  ROUGE-L:            {metrics['rougeL']:.4f}")
    print(f"  METEOR:             {metrics['meteor']:.4f}")
    print(f"  BERTScore F1:       {metrics['bertscore_f1']:.4f}")
    print(f"  Avg Inference Time: {metrics['avg_inference_time']:.2f}s")
    print(f"\n  Per-task accuracy:")
    for task, data in task_results.items():
        print(f"    {task}: {data['accuracy']:.4f} ({data['count']} samples)")

    # ============================================================
    # SAVE RESULTS
    # ============================================================
    print("\n" + "=" * 60)
    print("  SAVING RESULTS")
    print("=" * 60)

    # Save metrics JSON
    metrics_path = os.path.join(results_dir, f"metrics_{model_name.replace(' ', '_')}.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics: {metrics_path}")

    # Save detailed predictions CSV
    pred_df.to_csv(os.path.join(results_dir, "detailed_predictions.csv"), index=False)
    print(f"  Predictions: {results_dir}/detailed_predictions.csv")

    # Save per-task results CSV
    task_rows = []
    for task, data in task_results.items():
        task_rows.append({
            'Model': model_name,
            'Task': task,
            'Accuracy': data['accuracy'],
            'Count': data['count'],
        })
    task_df = pd.DataFrame(task_rows)
    task_df.to_csv(os.path.join(results_dir, "per_task_results.csv"), index=False)
    print(f"  Per-task: {results_dir}/per_task_results.csv")

    # Save benchmark summary CSV
    summary_df = pd.DataFrame([metrics], index=[model_name])
    summary_df.index.name = 'Model'
    summary_df.to_csv(os.path.join(results_dir, "benchmark_results.csv"))
    print(f"  Summary: {results_dir}/benchmark_results.csv")

    # Save text summary
    summary_path = os.path.join(results_dir, "benchmark_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Student Model Benchmark Results\n")
        f.write(f"{'=' * 60}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Base: {args.student_path}\n")
        f.write(f"Adapter: {args.adapter_path or 'None'}\n")
        f.write(f"Dataset: {args.dataset} [{args.split}] filter={args.filter_type}\n")
        f.write(f"Samples: {len(dataset)}\n\n")
        f.write(f"Metrics:\n")
        f.write(f"  MCQ Accuracy: {metrics['mcq_accuracy']:.4f}\n")
        f.write(f"  BLEU:         {metrics['bleu']:.4f}\n")
        f.write(f"  ROUGE-1:      {metrics['rouge1']:.4f}\n")
        f.write(f"  ROUGE-2:      {metrics['rouge2']:.4f}\n")
        f.write(f"  ROUGE-L:      {metrics['rougeL']:.4f}\n")
        f.write(f"  METEOR:       {metrics['meteor']:.4f}\n")
        f.write(f"  BERTScore F1: {metrics['bertscore_f1']:.4f}\n\n")
        f.write(f"Per-Task Accuracy:\n")
        for task, data in task_results.items():
            f.write(f"  {task}: {data['accuracy']:.4f} ({data['count']} samples)\n")
        f.write(f"\nTiming:\n")
        f.write(f"  Model Load:     {metrics['model_load_time']:.1f}s\n")
        f.write(f"  Avg Inference:  {metrics['avg_inference_time']:.2f}s/sample\n")
        f.write(f"  Total Inference:{metrics['total_inference_time']:.1f}s\n")
    print(f"  Summary text: {summary_path}")

    # Generate plots
    print("\n  Generating plots...")
    generate_plots(model_name, metrics, task_results, plots_dir)

    # Cleanup
    del model
    clear_gpu_memory()

    print("\n" + "=" * 60)
    print("  EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
