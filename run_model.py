"""
Evaluate a single VLM model on CV-Bench.

Usage:
    python run_model.py --config config.yaml --model InternVL2.5-1B
    python run_model.py --config config.yaml --model Gemma-4-E2B-it
    python run_model.py --config config.yaml --model Qwen3-VL-8B-4bit
"""

import argparse
import time
import warnings
from pathlib import Path
from io import BytesIO

import yaml
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from vlm_bench.utils import (
    clear_gpu_memory,
    extract_mcq_answer,
    extract_gt_answer,
    format_prompt,
)
from vlm_bench.models import get_inference_class
from vlm_bench.metrics import MetricsComputer
from vlm_bench.dataset import load_cvbench_data

warnings.filterwarnings("ignore")


def evaluate_model(config, model_config):
    """Run evaluation for a single model and save results."""
    model_name = model_config['name']
    model_path = model_config['path']
    model_type = model_config['type']

    results_dir = Path(config['output']['results_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print(f"EVALUATING: {model_name}")
    print(f"  Path: {model_path}")
    print(f"  Type: {model_type}")
    print("="*60)

    # Load dataset
    dataset = load_cvbench_data(config)

    # Load model
    print(f"\n  Loading model...")
    start_time = time.time()

    InferenceClass = get_inference_class(model_type)
    if InferenceClass is None:
        print(f"  Unknown model type: {model_type}")
        return None

    try:
        kwargs = {
            'model_path': model_path,
            'max_new_tokens': model_config.get('max_new_tokens', 128),
        }
        # Pass adapter_path for finetuned (LoRA) models
        if model_config.get('adapter_path'):
            kwargs['adapter_path'] = model_config['adapter_path']
            print(f"  Adapter: {model_config['adapter_path']}")

        if model_type in ("internvl", "internvl_lora"):
            kwargs['dtype'] = model_config.get('dtype', 'bfloat16')
        elif model_type in ("smolvlm", "smolvlm_lora", "sail", "spatialbot"):
            kwargs['dtype'] = model_config.get('dtype', 'bfloat16')
        elif model_type in ("qwen", "qwen_lora", "llama"):
            kwargs['load_in_4bit'] = model_config.get('load_in_4bit', True)

        print(f"  kwargs: {list(kwargs.keys())}")
        inference = InferenceClass(**kwargs)
    except Exception as e:
        print(f"  Failed to load model: {e}")
        return None

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

        prompt = format_prompt(question, choices)

        t0 = time.time()
        response = inference.generate(image, prompt)
        inference_times.append(time.time() - t0)

        pred_letter = extract_mcq_answer(response)
        gt_letter = extract_gt_answer(answer)

        predictions.append(response)
        pred_answers.append(pred_letter)
        gt_answers.append(gt_letter)
        references.append(answer)
        tasks.append(task)

        if (i + 1) % 25 == 0:
            running_acc = sum(1 for p, g in zip(pred_answers, gt_answers) if p == g) / len(pred_answers)
            avg_time = np.mean(inference_times)
            print(f"    [{i+1}/{len(dataset)}] Running Acc: {running_acc:.3f} | Avg time: {avg_time:.2f}s/sample")

    # Compute metrics
    print(f"\n  Computing metrics...")
    metrics_computer = MetricsComputer()
    results = metrics_computer.compute_all(predictions, references, pred_answers, gt_answers)
    results['avg_inference_time'] = np.mean(inference_times)
    results['total_inference_time'] = sum(inference_times)
    results['model_load_time'] = load_time

    # Per-task results
    pred_df = pd.DataFrame({
        'model': model_name,
        'task': tasks,
        'prediction': predictions,
        'pred_answer': pred_answers,
        'gt_answer': gt_answers,
        'reference': references,
    })

    task_results = metrics_computer.compute_per_task(pred_df)

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

    # Save model-specific results
    safe_name = model_name.replace("/", "_").replace(" ", "_")
    pred_df.to_csv(results_dir / f"predictions_{safe_name}.csv", index=False)

    results_series = pd.Series(results, name=model_name)
    results_series.to_json(results_dir / f"metrics_{safe_name}.json")

    task_df = pd.DataFrame([
        {'Model': model_name, 'Task': t, 'Accuracy': r['accuracy'], 'Samples': r['count']}
        for t, r in task_results.items()
    ])
    task_df.to_csv(results_dir / f"per_task_{safe_name}.csv", index=False)

    print(f"\n  Saved results to: {results_dir}/")
    print(f"    - predictions_{safe_name}.csv")
    print(f"    - metrics_{safe_name}.json")
    print(f"    - per_task_{safe_name}.csv")

    # Cleanup
    print(f"\n  Clearing {model_name} from GPU memory...")
    inference.cleanup()
    clear_gpu_memory()
    if torch.cuda.is_available():
        print(f"  GPU memory cleared. Free: {torch.cuda.mem_get_info()[0]/1e9:.2f} GB")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate a single VLM model on CV-Bench")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config YAML file')
    parser.add_argument('--model', type=str, required=True,
                        help='Name of the model to evaluate (must match a name in config.yaml)')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Find the requested model in config
    model_config = None
    for m in config['models']:
        if m['name'] == args.model:
            model_config = m
            break

    if model_config is None:
        available = [m['name'] for m in config['models']]
        print(f"Error: Model '{args.model}' not found in config.")
        print(f"Available models: {available}")
        return

    print("\n" + "="*60)
    print("VLM SPATIAL UNDERSTANDING BENCHMARK — SINGLE MODEL")
    print("="*60)
    print(f"  Config: {args.config}")
    print(f"  Model:  {args.model}")
    print(f"  Dataset: {config['dataset']['name']}")
    print(f"  Samples: {config['dataset']['num_samples']}")
    print(f"  Type filter: {config['dataset']['type_filter']}")
    print(f"  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"  VRAM: {torch.cuda.mem_get_info()[1]/1e9:.1f} GB")

    evaluate_model(config, model_config)

    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
