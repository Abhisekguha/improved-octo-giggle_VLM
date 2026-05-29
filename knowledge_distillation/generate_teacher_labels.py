"""
Step 1: Generate teacher labels from finetuned Qwen3-VL-8B.

Runs the teacher on CV-Bench 2D test set and saves:
  - Hard predictions (answer letter)
  - Soft labels (probability distribution over options)
  - Rationales (chain-of-thought reasoning)
  - Visual features (intermediate ViT hidden states for feature distillation)
"""

import argparse
import os
import numpy as np
import torch
from tqdm import tqdm

# Support both direct execution and module execution
try:
    from knowledge_distillation.config import TeacherConfig, DataConfig, PathConfig
    from knowledge_distillation.utils import (
        extract_mcq_answer,
        format_question,
        load_cvbench,
        load_image,
        save_jsonl,
    )
except ModuleNotFoundError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from knowledge_distillation.config import TeacherConfig, DataConfig, PathConfig
    from knowledge_distillation.utils import (
        extract_mcq_answer,
        format_question,
        load_cvbench,
        load_image,
        save_jsonl,
    )


# ------------------------------------------------------------------
# Model loading
# ------------------------------------------------------------------

def load_teacher(cfg: TeacherConfig):
    """Load the finetuned Qwen3-VL teacher model."""
    from unsloth import FastVisionModel

    model, tokenizer = FastVisionModel.from_pretrained(
        cfg.model_path,
        load_in_4bit=cfg.load_in_4bit,
        dtype=torch.bfloat16 if cfg.dtype == "bfloat16" else torch.float16,
    )

    if cfg.adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, cfg.adapter_path)
        model = model.merge_and_unload()

    FastVisionModel.for_inference(model)
    return model, tokenizer


# ------------------------------------------------------------------
# Visual feature extraction
# ------------------------------------------------------------------

class VisualFeatureExtractor:
    """Hook-based extractor for intermediate ViT hidden states."""

    def __init__(self, model, layer_indices: list):
        self.features = {}
        self.hooks = []
        self._register_hooks(model, layer_indices)

    def _find_vision_encoder(self, model):
        """Find the vision encoder module in the model hierarchy."""
        # Qwen3-VL uses model.visual or model.model.visual
        for attr in ["visual", "vision_model", "vision_tower", "vit"]:
            if hasattr(model, attr):
                return getattr(model, attr)
            if hasattr(model, "model") and hasattr(model.model, attr):
                return getattr(model.model, attr)
        # Try deeper: model.model.model.visual (for wrapped models)
        if hasattr(model, "model") and hasattr(model.model, "model"):
            for attr in ["visual", "vision_model", "vision_tower"]:
                if hasattr(model.model.model, attr):
                    return getattr(model.model.model, attr)
        return None

    def _get_encoder_layers(self, vision_module):
        """Get the list of transformer layers from the vision encoder."""
        # Try common layer container names
        for attr in ["blocks", "layers", "encoder.layers", "encoder.layer"]:
            parts = attr.split(".")
            mod = vision_module
            for p in parts:
                if hasattr(mod, p):
                    mod = getattr(mod, p)
                else:
                    mod = None
                    break
            if mod is not None and hasattr(mod, "__len__"):
                return list(mod)
        return []

    def _register_hooks(self, model, layer_indices):
        """Register forward hooks on specified ViT layers."""
        vision = self._find_vision_encoder(model)
        if vision is None:
            print("  WARNING: Could not find vision encoder — skipping feature extraction")
            return

        layers = self._get_encoder_layers(vision)
        if not layers:
            print("  WARNING: Could not find ViT layers — skipping feature extraction")
            return

        num_layers = len(layers)
        for idx in layer_indices:
            actual_idx = idx if idx >= 0 else num_layers + idx
            if 0 <= actual_idx < num_layers:
                layer = layers[actual_idx]
                hook = layer.register_forward_hook(self._make_hook(actual_idx))
                self.hooks.append(hook)

        print(f"  Registered feature hooks on ViT layers: {layer_indices} "
              f"(resolved to {[i if i >= 0 else num_layers + i for i in layer_indices]})")

    def _make_hook(self, layer_idx):
        def hook_fn(module, input, output):
            # output can be a tensor or tuple
            feat = output[0] if isinstance(output, tuple) else output
            self.features[layer_idx] = feat.detach().cpu()
        return hook_fn

    def get_features(self):
        """Return extracted features and clear buffer."""
        feats = dict(self.features)
        self.features.clear()
        return feats

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


# ------------------------------------------------------------------
# Inference helpers
# ------------------------------------------------------------------

def _get_soft_labels(scores, tokenizer, num_options: int):
    """Extract soft-label distribution from first-token logits."""
    if not scores:
        return None
    first_logits = scores[0][0]
    # tokenizer may be a Processor; get the underlying tokenizer for encode()
    _tok = tokenizer.tokenizer if hasattr(tokenizer, 'tokenizer') else tokenizer
    option_ids = [
        _tok.encode(chr(65 + i), add_special_tokens=False)[0]
        for i in range(num_options)
    ]
    option_logits = first_logits[option_ids]
    return torch.softmax(option_logits, dim=-1).cpu().tolist()


def _generate_rationale(model, tokenizer, image, formatted_q, cfg: TeacherConfig):
    """Generate step-by-step rationale from the teacher."""
    messages = [
        {"role": "system", "content": [{"type": "text", "text":
            "You are an expert at spatial reasoning. Think step by step."}]},
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": (
                f"{formatted_q}\n\n"
                "Explain your reasoning step by step, then give the answer."
            )},
        ]},
    ]

    input_text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True
    )
    inputs = tokenizer(
        image, input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=cfg.rationale_max_tokens,
            temperature=cfg.rationale_temperature,
            do_sample=True,
        )

    gen_ids = outputs[:, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()


# ------------------------------------------------------------------
# Main generation loop
# ------------------------------------------------------------------

def generate_labels(model, tokenizer, dataset, cfg: TeacherConfig, output_path: str,
                    features_dir: str = None):
    """Run teacher inference and collect labels + visual features."""
    results = []

    # Setup feature extraction if requested
    extractor = None
    if cfg.save_visual_features and features_dir:
        extractor = VisualFeatureExtractor(model, cfg.visual_feature_layers)
        os.makedirs(features_dir, exist_ok=True)

    for idx in tqdm(range(len(dataset)), desc="Teacher inference"):
        sample = dataset[idx]
        image = load_image(sample["image"])
        question = sample["question"]
        options = sample.get("options", sample.get("choices"))
        answer = sample["answer"]
        task = sample.get("task", sample.get("type", "unknown"))

        formatted_q = format_question(question, options)

        # --- Direct answer with logits ---
        messages = [
            {"role": "system", "content": [{"type": "text", "text":
                "You are an expert at spatial reasoning and visual understanding. "
                "Answer the multiple choice question about the image."}]},
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": formatted_q},
            ]},
        ]

        input_text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True
        )
        inputs = tokenizer(
            image, input_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=1.0,
                do_sample=False,
                output_scores=cfg.save_logits,
                return_dict_in_generate=True,
            )

        gen_ids = outputs.sequences[:, inputs["input_ids"].shape[1]:]
        prediction = tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()

        # Soft labels
        num_options = len(options) if isinstance(options, list) else 4
        soft_labels = (
            _get_soft_labels(outputs.scores, tokenizer, num_options)
            if cfg.save_logits else None
        )

        # Save visual features (extracted via hooks during forward pass)
        if extractor and extractor.hooks:
            feats = extractor.get_features()
            if feats:
                feat_path = os.path.join(features_dir, f"feat_{idx:05d}.npz")
                np.savez_compressed(
                    feat_path,
                    **{f"layer_{k}": v.float().numpy() for k, v in feats.items()}
                )

        # Rationale (skip if not needed — faster)
        rationale = None
        if cfg.generate_rationale:
            rationale = _generate_rationale(model, tokenizer, image, formatted_q, cfg)

        results.append({
            "idx": idx,
            "question": question,
            "options": options if isinstance(options, list) else eval(options),
            "ground_truth": answer,
            "teacher_prediction": prediction,
            "teacher_soft_labels": soft_labels,
            "teacher_rationale": rationale,
            "task": task,
        })

        # Periodic checkpoint
        if (idx + 1) % 50 == 0:
            save_jsonl(results, output_path)

    if extractor:
        extractor.remove_hooks()

    return results


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate teacher labels for KD")
    parser.add_argument("--model_path", type=str, default="abhi26/subCV_qwen3-8B_lora")
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="nyu-visionx/CV-Bench")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output", type=str,
                        default="knowledge_distillation/teacher_outputs/teacher_labels.jsonl")
    parser.add_argument("--features_dir", type=str,
                        default="knowledge_distillation/teacher_outputs/visual_features")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--no_rationale", action="store_true")
    parser.add_argument("--no_logits", action="store_true")
    parser.add_argument("--no_features", action="store_true", help="Skip visual feature extraction (faster)")
    args = parser.parse_args()

    teacher_cfg = TeacherConfig(
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        generate_rationale=not args.no_rationale,
        save_logits=not args.no_logits,
        save_visual_features=not args.no_features,
    )

    print("=" * 60)
    print("  STEP 1: Generate Teacher Labels")
    print("=" * 60)
    print(f"  Model    : {teacher_cfg.model_path}")
    print(f"  Dataset  : {args.dataset} [{args.split}] → 2D only")
    print(f"  Output   : {args.output}")
    print(f"  Rationale: {teacher_cfg.generate_rationale}")
    print(f"  Logits   : {teacher_cfg.save_logits}")
    print("=" * 60)

    # Load dataset (2D test split)
    dataset = load_cvbench(
        args.dataset, args.split, filter_type="2D", max_samples=args.max_samples
    )
    print(f"Samples: {len(dataset)}")

    # Load teacher
    model, tokenizer = load_teacher(teacher_cfg)
    print("Teacher model loaded.\n")

    # Generate labels
    features_dir = args.features_dir if teacher_cfg.save_visual_features else None
    results = generate_labels(model, tokenizer, dataset, teacher_cfg, args.output,
                              features_dir=features_dir)
    save_jsonl(results, args.output)

    # Report
    correct = sum(
        1 for r in results
        if extract_mcq_answer(r["teacher_prediction"]) == extract_mcq_answer(str(r["ground_truth"]))
    )
    print(f"\nTeacher accuracy: {correct}/{len(results)} = {correct/len(results)*100:.1f}%")
    print(f"Labels saved to: {args.output}")


if __name__ == "__main__":
    main()
