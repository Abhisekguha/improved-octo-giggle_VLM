"""
Step 3: Evaluate the distilled student model on CV-Bench 2D test set.

Loads the student (InternVL3.5-1B + LoRA adapter) and reports accuracy
overall and per-task.
"""

import argparse
import json
import os
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
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

    # Find closest aspect ratio
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
                # Preprocess image to pixel_values
                pixel_values = load_image(image, max_num=12).to(torch.bfloat16)
                if torch.cuda.is_available():
                    pixel_values = pixel_values.cuda()
                
                generation_config = {"max_new_tokens": 10, "do_sample": False}
                # InternVL chat signature: chat(tokenizer, pixel_values, question, generation_config)
                response = model.chat(tokenizer, pixel_values, query, generation_config)
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
