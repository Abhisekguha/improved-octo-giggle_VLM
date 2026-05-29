"""Shared utility functions for VLM benchmarking."""

import gc
import re
import json

import torch


# Spatial understanding system prompt injected into all models
SPATIAL_SYSTEM_PROMPT = (
    """You are a visual spatial reasoning expert.
    Given an image and a multiple-choice question about spatial relationships 
    (object positions, left/right/above/below, depth, distance to camera, counting), 
    look at the image carefully, answer only the correct option.
    Do not explain."""
)


def clear_gpu_memory():
    """Aggressively clear GPU memory between model evaluations."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def extract_mcq_answer(text):
    """Extract MCQ answer from model output (supports A-Z and numeric options)."""
    text = text.strip()
    # Try parenthesized letter like (A), (B), ..., (Z)
    match = re.search(r'\(([A-Za-z])\)', text)
    if match:
        return match.group(1).upper()
    # Try parenthesized number like (1), (2)
    match = re.search(r'\((\d+)\)', text)
    if match:
        return match.group(1)
    # Try standalone letter/number at start
    match = re.match(r'^([A-Za-z\d])\b', text)
    if match:
        return match.group(1).upper()
    # Try letter/number followed by punctuation
    match = re.search(r'\b([A-Za-z\d])[\.:\)\s]', text)
    if match:
        return match.group(1).upper()
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

    prompt = f"{question}\n\nOptions:\n{options}\n\nAnswer with only the letter of the correct option."
    return prompt


def extract_gt_answer(answer):
    """Extract ground truth answer letter from dataset answer field."""
    if isinstance(answer, str):
        match = re.search(r'\(([A-Za-z])\)', answer)
        if match:
            return match.group(1).upper()
        match = re.match(r'^([A-Za-z])$', answer.strip())
        if match:
            return match.group(1).upper()
        return answer.strip()[0].upper() if answer.strip() else ""
    return str(answer)

