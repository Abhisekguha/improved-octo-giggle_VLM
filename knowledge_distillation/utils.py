"""Shared utilities for the Knowledge Distillation pipeline."""

import json
import os
import sys
from pathlib import Path
from typing import List, Union

from PIL import Image

# Ensure parent is importable
sys.path.insert(0, str(Path(__file__).parent.parent))
from vlm_bench.utils import extract_mcq_answer  # noqa: E402


def format_question(question: str, options) -> str:
    """Format a question with MCQ options (A, B, C, ...)."""
    if isinstance(options, str):
        try:
            options = json.loads(options)
        except (json.JSONDecodeError, TypeError):
            import ast
            options = ast.literal_eval(options)

    if isinstance(options, list):
        options_text = "\n".join(
            f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options)
        )
    elif isinstance(options, dict):
        options_text = "\n".join(f"{k}. {v}" for k, v in options.items())
    else:
        options_text = str(options)

    return (
        f"Question: {question}\n"
        f"Options:\n{options_text}\n\n"
        f"Answer with the correct option letter."
    )


def load_jsonl(path: str) -> List[dict]:
    """Load a JSONL file into a list of dicts."""
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def save_jsonl(data: List[dict], path: str):
    """Save a list of dicts to a JSONL file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_image(image) -> Image.Image:
    """Normalize an image input to a PIL Image."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, str):
        return Image.open(image).convert("RGB")
    return image


def load_cvbench(dataset_name: str, split: str, filter_type: str = "2D", max_samples: int = None):
    """Load and filter CV-Bench dataset."""
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split=split)

    # Filter to specified type (2D spatial tasks)
    if "type" in ds.column_names and filter_type:
        ds = ds.filter(lambda x: x["type"].upper() == filter_type.upper())

    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    return ds
