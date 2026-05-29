"""
Step 3: Evaluate the distilled student model on CV-Bench 2D test set.

Loads the student (InternVL3.5-1B + LoRA adapter) and reports accuracy
overall and per-task.
"""

import argparse
import json
import os
import torch
from tqdm import tqdm

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
# Model loading
# ------------------------------------------------------------------

def load_student(model_path: str, adapter_path: str = None):
    """Load the distilled student model (base + optional LoRA adapter)."""
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
    return model, tokenizer


# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------

def evaluate(model, tokenizer, dataset):
    """Evaluate model on CV-Bench and return metrics."""
    correct = 0
    total = 0
    results_by_task = {}
    predictions = []

    for idx in tqdm(range(len(dataset)), desc="Evaluating"):
        sample = dataset[idx]
        image = sample["image"]
        question = sample["question"]
        options = sample.get("choices", sample.get("options", []))
        answer = sample["answer"]
        task = sample.get("task", sample.get("type", "unknown"))

        formatted_q = format_question(question, options)
        query = f"<image>\n{formatted_q}"

        # Generate prediction
        try:
            if hasattr(model, "chat"):
                response = model.chat(
                    tokenizer=tokenizer,
                    pixel_values=None,
                    question=query,
                    generation_config={"max_new_tokens": 10, "do_sample": False},
                    images=[image],
                )
            else:
                inputs = tokenizer(query, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=10, do_sample=False)
                response = tokenizer.decode(
                    out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
                )
        except Exception as e:
            response = ""
            print(f"  [Error idx={idx}]: {e}")

        pred_letter = extract_mcq_answer(response)
        gt_letter = extract_mcq_answer(str(answer))
        is_correct = pred_letter == gt_letter

        correct += int(is_correct)
        total += 1

        if task not in results_by_task:
            results_by_task[task] = {"correct": 0, "total": 0}
        results_by_task[task]["correct"] += int(is_correct)
        results_by_task[task]["total"] += 1

        predictions.append({
            "idx": idx,
            "question": question,
            "ground_truth": gt_letter,
            "prediction": pred_letter,
            "correct": is_correct,
            "task": task,
        })

    accuracy = correct / total if total > 0 else 0.0
    per_task = {
        t: v["correct"] / v["total"] for t, v in results_by_task.items()
    }

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "per_task": per_task,
        "predictions": predictions,
    }


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate distilled student model")
    parser.add_argument("--student_path", type=str, default="OpenGVLab/InternVL3_5-1B-Instruct")
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="nyu-visionx/CV-Bench")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output", type=str,
                        default="knowledge_distillation/eval_results/student_eval.json")
    args = parser.parse_args()

    print("=" * 60)
    print("  STEP 3: Evaluate Distilled Student")
    print("=" * 60)
    print(f"  Model   : {args.student_path}")
    print(f"  Adapter : {args.adapter_path or 'None (baseline)'}")
    print(f"  Dataset : {args.dataset} [{args.split}] → 2D only")
    print("=" * 60)

    # Load dataset
    dataset = load_cvbench(
        args.dataset, args.split, filter_type="2D", max_samples=args.max_samples
    )
    print(f"Evaluation samples: {len(dataset)}")

    # Load model
    model, tokenizer = load_student(args.student_path, args.adapter_path)

    # Evaluate
    results = evaluate(model, tokenizer, dataset)

    # Print results
    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"  Overall Accuracy: {results['accuracy']*100:.2f}%")
    print(f"  Correct: {results['correct']}/{results['total']}")
    print("\n  Per-task:")
    for task, acc in results["per_task"].items():
        print(f"    {task}: {acc*100:.2f}%")

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({k: v for k, v in results.items() if k != "predictions"}, f, indent=2)

    pred_path = args.output.replace(".json", "_predictions.jsonl")
    with open(pred_path, "w") as f:
        for p in results["predictions"]:
            f.write(json.dumps(p) + "\n")

    print(f"\n  Results saved to: {args.output}")
    print(f"  Predictions saved to: {pred_path}")


if __name__ == "__main__":
    main()
