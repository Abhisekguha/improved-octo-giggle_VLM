"""
Step 2b: Train SmolVLM-Instruct student using teacher-generated labels.

Supports two KD modes:
  - response: SFT on teacher answers/rationales (simpler, effective)
  - soft_label: KL-divergence loss on teacher probability distributions

Enhanced with:
  - Vision encoder LoRA (trains spatial feature extraction)
  - Projector unfreezing (adapts vision→language bridge)
  - Feature distillation (MSE loss on intermediate ViT hidden states)

Student: HuggingFaceTB/SmolVLM-Instruct
"""

import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use single GPU to avoid DataParallel issues
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional
from PIL import Image

from datasets import load_dataset
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model

# Support both direct execution and module execution
try:
    from knowledge_distillation.config import KDConfig, DataConfig, PathConfig
    from knowledge_distillation.utils import (
        extract_mcq_answer,
        format_question,
        load_cvbench,
        load_jsonl,
    )
except ModuleNotFoundError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from knowledge_distillation.config import KDConfig, DataConfig, PathConfig
    from knowledge_distillation.utils import (
        extract_mcq_answer,
        format_question,
        load_cvbench,
        load_jsonl,
    )


# ------------------------------------------------------------------
# SmolVLM Student Config
# ------------------------------------------------------------------

from dataclasses import dataclass, field


@dataclass
class SmolVLMStudentConfig:
    """Student model configuration for SmolVLM-Instruct."""
    model_path: str = "HuggingFaceTB/SmolVLM-Instruct"
    # LLM LoRA
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    # Vision encoder LoRA
    vision_lora: bool = True
    vision_lora_r: int = 8
    vision_lora_alpha: int = 8
    vision_lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "out_proj",
        "fc1", "fc2",
    ])
    # Projector training
    train_projector: bool = True


# ------------------------------------------------------------------
# Dataset preparation
# ------------------------------------------------------------------

def build_training_data(
    teacher_labels: List[Dict],
    cv_bench_dataset,
    kd_cfg: KDConfig,
    features_dir: Optional[str] = None,
) -> List[Dict]:
    """Combine teacher labels with dataset images into training items."""
    training_data = []

    for item in teacher_labels:
        idx = item["idx"]
        if idx >= len(cv_bench_dataset):
            continue

        image = cv_bench_dataset[idx]["image"]
        formatted_q = format_question(item["question"], item["options"])

        # Determine target text
        if kd_cfg.use_rationale and item.get("teacher_rationale"):
            target = item["teacher_rationale"]
            pred = extract_mcq_answer(item["teacher_prediction"])
            if pred and pred not in target[-20:]:
                target += f"\n\nAnswer: {pred}"
        else:
            target = item["teacher_prediction"].strip()

        # Feature path for feature distillation
        feat_path = None
        if features_dir and kd_cfg.feature_distillation:
            candidate = os.path.join(features_dir, f"feat_{idx:05d}.npz")
            if os.path.exists(candidate):
                feat_path = candidate

        training_data.append({
            "image": image,
            "question": formatted_q,
            "answer": target,
            "soft_labels": item.get("teacher_soft_labels"),
            "ground_truth": item.get("ground_truth"),
            "task": item.get("task", "unknown"),
            "teacher_feat_path": feat_path,
        })

    return training_data


# ------------------------------------------------------------------
# Data collator for SmolVLM
# ------------------------------------------------------------------

class SmolVLMKDCollator:
    """Collates batches for SmolVLM with optional soft-label and feature distillation support."""

    def __init__(self, processor, max_length: int = 2048, kd_mode: str = "response",
                 feature_distillation: bool = False):
        self.processor = processor
        self.max_length = max_length
        self.kd_mode = kd_mode
        self.feature_distillation = feature_distillation

    def __call__(self, batch):
        input_ids_list = []
        attention_mask_list = []
        labels_list = []
        pixel_values_list = []
        pixel_attention_mask_list = []
        soft_labels_list = []
        teacher_features_list = []

        for item in batch:
            image = item["image"]
            if hasattr(image, "convert"):
                image = image.convert("RGB")

            question = item["question"]
            answer = item["answer"]

            # Build chat messages for SmolVLM
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": question},
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": answer},
                    ]
                },
            ]

            # Apply chat template
            text = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=False,
                tokenize=False,
            )

            # Process with image (no truncation to avoid image token mismatch)
            inputs = self.processor(
                text=text,
                images=[image],
                return_tensors="pt",
                padding=False,
                truncation=False,
            )

            # Manually truncate input_ids/attention_mask AFTER processing
            # to preserve image token alignment
            if inputs["input_ids"].shape[-1] > self.max_length:
                inputs["input_ids"] = inputs["input_ids"][:, :self.max_length]
                inputs["attention_mask"] = inputs["attention_mask"][:, :self.max_length]

            input_ids = inputs["input_ids"].squeeze(0)
            attention_mask = inputs["attention_mask"].squeeze(0)
            pixel_values = inputs.get("pixel_values")
            pixel_attention_mask = inputs.get("pixel_attention_mask")

            # Build labels: mask everything before assistant response
            labels = input_ids.clone()
            # Find the assistant response start token
            # SmolVLM uses "Assistant:" or similar pattern
            tokenizer = self.processor.tokenizer if hasattr(self.processor, 'tokenizer') else self.processor
            # Encode the answer to find where it starts
            answer_tokens = tokenizer.encode(answer, add_special_tokens=False)
            # Find the position of answer tokens in input_ids
            input_list = input_ids.tolist()
            answer_start = -1
            for i in range(len(input_list) - len(answer_tokens) + 1):
                if input_list[i:i + len(answer_tokens)] == answer_tokens:
                    answer_start = i
                    break

            if answer_start > 0:
                labels[:answer_start] = -100
            else:
                # Fallback: mask first 80% as input
                mask_len = int(len(input_ids) * 0.8)
                labels[:mask_len] = -100

            labels[attention_mask == 0] = -100

            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            labels_list.append(labels)
            if pixel_values is not None:
                pixel_values_list.append(pixel_values.squeeze(0))
            if pixel_attention_mask is not None:
                pixel_attention_mask_list.append(pixel_attention_mask.squeeze(0))

            if item.get("soft_labels"):
                soft_labels_list.append(torch.tensor(item["soft_labels"]))

            # Load pre-computed teacher features
            if self.feature_distillation and item.get("teacher_feat_path"):
                feat_data = np.load(item["teacher_feat_path"])
                teacher_features_list.append(
                    {k: torch.from_numpy(v) for k, v in feat_data.items()}
                )
            elif self.feature_distillation:
                teacher_features_list.append(None)

        # Pad sequences
        max_seq_len = max(ids.shape[0] for ids in input_ids_list)
        pad_token_id = self.processor.tokenizer.pad_token_id if hasattr(self.processor, 'tokenizer') else 0
        if pad_token_id is None:
            pad_token_id = 0

        padded_input_ids, padded_attention_mask, padded_labels = [], [], []
        for ids, mask, lbl in zip(input_ids_list, attention_mask_list, labels_list):
            pad_len = max_seq_len - ids.shape[0]
            if pad_len > 0:
                ids = F.pad(ids, (0, pad_len), value=pad_token_id)
                mask = F.pad(mask, (0, pad_len), value=0)
                lbl = F.pad(lbl, (0, pad_len), value=-100)
            padded_input_ids.append(ids)
            padded_attention_mask.append(mask)
            padded_labels.append(lbl)

        result = {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(padded_attention_mask),
            "labels": torch.stack(padded_labels),
        }

        # Handle pixel values (may have different shapes per image due to dynamic patching)
        if pixel_values_list:
            # Pad pixel values to same shape
            max_patches = max(pv.shape[0] for pv in pixel_values_list)
            padded_pvs = []
            padded_pam = []
            for i, pv in enumerate(pixel_values_list):
                pad_patches = max_patches - pv.shape[0]
                if pad_patches > 0:
                    pv = F.pad(pv, (0, 0, 0, 0, 0, 0, 0, pad_patches), value=0.0)
                padded_pvs.append(pv)
                if pixel_attention_mask_list:
                    pam = pixel_attention_mask_list[i]
                    if pam.shape[0] < max_patches:
                        pam = F.pad(pam, (0, 0, 0, 0, 0, max_patches - pam.shape[0]), value=0)
                    padded_pam.append(pam)
            result["pixel_values"] = torch.stack(padded_pvs)
            if padded_pam:
                result["pixel_attention_mask"] = torch.stack(padded_pam)

        if soft_labels_list and self.kd_mode == "soft_label":
            max_opts = max(s.shape[0] for s in soft_labels_list)
            padded = []
            for s in soft_labels_list:
                if s.shape[0] < max_opts:
                    s = F.pad(s, (0, max_opts - s.shape[0]), value=0.0)
                padded.append(s)
            result["soft_labels"] = torch.stack(padded)

        if teacher_features_list and any(f is not None for f in teacher_features_list):
            result["teacher_features"] = teacher_features_list

        return result


# ------------------------------------------------------------------
# Feature Alignment Module
# ------------------------------------------------------------------

class FeatureAlignmentProjector(nn.Module):
    """Projects teacher/student features to a shared dim for MSE alignment."""

    def __init__(self, teacher_dim: int, student_dim: int, proj_dim: int = 256):
        super().__init__()
        self.teacher_proj = nn.Linear(teacher_dim, proj_dim)
        self.student_proj = nn.Linear(student_dim, proj_dim)

    def forward(self, teacher_feat, student_feat):
        t = self.teacher_proj(teacher_feat.float())
        s = self.student_proj(student_feat.float())
        return F.mse_loss(s, t.detach())


# ------------------------------------------------------------------
# Custom KD Trainer (soft-label + feature distillation)
# ------------------------------------------------------------------

class SmolVLMKDTrainer(Trainer):
    """Trainer with KL-divergence + feature distillation loss for SmolVLM."""

    def __init__(self, *args, kd_cfg: KDConfig = None,
                 feature_projectors: nn.ModuleDict = None,
                 vision_hook_features: dict = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.kd_cfg = kd_cfg or KDConfig()
        self.feature_projectors = feature_projectors
        self.vision_hook_features = vision_hook_features

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Extract custom inputs
        soft_labels = inputs.pop("soft_labels", None)
        teacher_features = inputs.pop("teacher_features", None)

        labels = inputs.get("labels")

        # SmolVLM accepts: input_ids, attention_mask, labels, pixel_values, pixel_attention_mask
        valid_keys = {"input_ids", "attention_mask", "labels", "pixel_values", "pixel_attention_mask"}
        filtered_inputs = {k: v for k, v in inputs.items() if k in valid_keys}

        # Cast pixel_values to model dtype
        if "pixel_values" in filtered_inputs:
            model_dtype = next(model.parameters()).dtype
            filtered_inputs["pixel_values"] = filtered_inputs["pixel_values"].to(dtype=model_dtype)

        outputs = model(**filtered_inputs)
        ce_loss = outputs.loss

        total_loss = ce_loss

        # --- Soft-label KL loss ---
        if soft_labels is not None and self.kd_cfg.kd_mode == "soft_label":
            logits = outputs.logits
            num_options = soft_labels.shape[-1]
            _tokenizer = self.processing_class if hasattr(self, 'processing_class') and self.processing_class else None
            if _tokenizer is None:
                _tokenizer = self.tokenizer
            option_ids = [
                _tokenizer.encode(chr(65 + i), add_special_tokens=False)[0]
                for i in range(num_options)
            ]

            batch_kd_loss = []

            for b in range(logits.shape[0]):
                non_masked = (labels[b] != -100).nonzero(as_tuple=True)[0]
                if len(non_masked) == 0:
                    continue
                pos = non_masked[0]
                student_logits = logits[b, pos - 1, option_ids]

                T = self.kd_cfg.temperature
                student_log_probs = F.log_softmax(student_logits / T, dim=-1)
                teacher_probs = F.softmax(soft_labels[b] / T, dim=-1)

                kd_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (T ** 2)
                batch_kd_loss.append(kd_loss)

            if batch_kd_loss:
                kd_loss = torch.stack(batch_kd_loss).mean()
                alpha = self.kd_cfg.alpha
                total_loss = alpha * kd_loss + (1 - alpha) * ce_loss

        # --- Feature distillation MSE loss ---
        if (self.kd_cfg.feature_distillation and teacher_features is not None
                and self.feature_projectors and self.vision_hook_features):
            feat_losses = []
            for b_idx, t_feats in enumerate(teacher_features):
                if t_feats is None:
                    continue
                for layer_key, teacher_feat in t_feats.items():
                    if layer_key not in self.vision_hook_features:
                        continue
                    student_feat = self.vision_hook_features[layer_key]
                    if student_feat is None:
                        continue
                    s_feat = student_feat[b_idx] if student_feat.dim() > 2 else student_feat
                    t_feat = teacher_feat.squeeze(0).to(s_feat.device)
                    if s_feat.shape[0] != t_feat.shape[0]:
                        min_len = min(s_feat.shape[0], t_feat.shape[0])
                        s_feat = s_feat[:min_len]
                        t_feat = t_feat[:min_len]
                    if layer_key in self.feature_projectors:
                        loss = self.feature_projectors[layer_key](t_feat, s_feat)
                        feat_losses.append(loss)

            if feat_losses:
                feat_loss = torch.stack(feat_losses).mean()
                total_loss = total_loss + self.kd_cfg.feature_loss_weight * feat_loss

        # Clear hook buffer
        if self.vision_hook_features:
            self.vision_hook_features.clear()

        return (total_loss, outputs) if return_outputs else total_loss


# ------------------------------------------------------------------
# Model setup with Vision LoRA + Projector Unfreezing
# ------------------------------------------------------------------

def _find_vision_encoder(model):
    """Find the vision encoder in SmolVLM model."""
    # SmolVLM uses 'vision_model' or 'model.vision_model'
    for attr_path in ["vision_model", "model.vision_model", "visual", "vit", "vision_tower"]:
        parts = attr_path.split(".")
        mod = model
        for p in parts:
            if hasattr(mod, p):
                mod = getattr(mod, p)
            else:
                mod = None
                break
        if mod is not None and mod is not model:
            return mod, attr_path
    return None, None


def _find_projector(model):
    """Find the vision-language projector/bridge in SmolVLM."""
    for attr_path in ["multi_modal_projector", "model.multi_modal_projector",
                      "connector", "model.connector", "mm_projector",
                      "vision_proj", "bridge", "mlp1"]:
        parts = attr_path.split(".")
        mod = model
        for p in parts:
            if hasattr(mod, p):
                mod = getattr(mod, p)
            else:
                mod = None
                break
        if mod is not None and mod is not model:
            return mod, attr_path
    return None, None


def _get_vision_layers(vision_module):
    """Get list of transformer blocks from the vision encoder."""
    for attr_path in ["encoder.layers", "layers", "blocks", "encoder.layer"]:
        parts = attr_path.split(".")
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


def setup_student(student_cfg: SmolVLMStudentConfig, kd_cfg: KDConfig):
    """Load SmolVLM student model with LLM LoRA + Vision LoRA + unfrozen projector."""
    model = AutoModelForImageTextToText.from_pretrained(
        student_cfg.model_path,
        torch_dtype=torch.bfloat16 if kd_cfg.bf16 else torch.float16,
        trust_remote_code=True,
        device_map={"": 0},
    )
    processor = AutoProcessor.from_pretrained(
        student_cfg.model_path, trust_remote_code=True
    )

    # ---- 1. Apply LoRA to LLM backbone ----
    model_module_names = [name for name, _ in model.named_modules()]
    valid_llm_targets = [
        t for t in student_cfg.lora_target_modules
        if any(t in m for m in model_module_names)
    ]
    if not valid_llm_targets:
        valid_llm_targets = ["q_proj", "k_proj", "v_proj", "o_proj"]

    lora_config = LoraConfig(
        r=student_cfg.lora_r,
        lora_alpha=student_cfg.lora_alpha,
        lora_dropout=student_cfg.lora_dropout,
        target_modules=valid_llm_targets,
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    # Enable gradient checkpointing to save VRAM (SmolVLM/Idefics3 supports it)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    print(f"\n  [LLM LoRA] targets: {valid_llm_targets}")

    # ---- 2. Apply LoRA to Vision Encoder ----
    base_model_ref = model.base_model.model if hasattr(model, 'base_model') else model
    vision_module, vision_attr = _find_vision_encoder(base_model_ref)
    if student_cfg.vision_lora and vision_module is not None:
        vision_module_names = [n for n, _ in vision_module.named_modules()]
        valid_vision_targets = [
            t for t in student_cfg.vision_lora_target_modules
            if any(t in m for m in vision_module_names)
        ]
        if valid_vision_targets:
            from peft import inject_adapter_in_model
            vision_lora_config = LoraConfig(
                r=student_cfg.vision_lora_r,
                lora_alpha=student_cfg.vision_lora_alpha,
                lora_dropout=student_cfg.lora_dropout,
                target_modules=valid_vision_targets,
                bias="none",
            )
            inject_adapter_in_model(vision_lora_config, vision_module, adapter_name="vision_lora")
            # Ensure vision LoRA params are trainable
            for name, param in vision_module.named_parameters():
                if "lora_" in name:
                    param.requires_grad = True
            print(f"  [Vision LoRA] targets: {valid_vision_targets}")
        else:
            print(f"  [Vision LoRA] WARNING: no matching modules in {vision_attr}")
    elif vision_module is None:
        print("  [Vision LoRA] WARNING: vision encoder not found")

    # ---- 3. Unfreeze Projector ----
    projector, proj_attr = _find_projector(base_model_ref)
    if student_cfg.train_projector and projector is not None:
        for param in projector.parameters():
            param.requires_grad = True
        proj_params = sum(p.numel() for p in projector.parameters() if p.requires_grad)
        print(f"  [Projector] unfrozen '{proj_attr}' — {proj_params:,} trainable params")
    elif projector is None:
        print("  [Projector] WARNING: projector module not found")

    # Print summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Total params:     {total_params:,}")
    print(f"  Trainable params: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")

    return model, processor


# ------------------------------------------------------------------
# Vision feature hooks for student
# ------------------------------------------------------------------

def setup_student_vision_hooks(model, kd_cfg: KDConfig):
    """Register hooks on student's vision encoder to capture intermediate features."""
    hook_features = {}
    hooks = []

    base_model_ref = model.base_model.model if hasattr(model, "base_model") else model
    vision_module, _ = _find_vision_encoder(base_model_ref)
    if vision_module is None:
        return hook_features, hooks

    layers = _get_vision_layers(vision_module)
    if not layers:
        return hook_features, hooks

    num_layers = len(layers)
    for idx in kd_cfg.feature_layers:
        actual_idx = idx if idx >= 0 else num_layers + idx
        if 0 <= actual_idx < num_layers:
            layer_key = f"layer_{actual_idx}"

            def make_hook(key):
                def hook_fn(module, input, output):
                    feat = output[0] if isinstance(output, tuple) else output
                    hook_features[key] = feat
                return hook_fn

            hook = layers[actual_idx].register_forward_hook(make_hook(layer_key))
            hooks.append(hook)

    print(f"  [Feature KD] hooks on student ViT layers: {kd_cfg.feature_layers}")
    return hook_features, hooks


# ------------------------------------------------------------------
# Training entry point
# ------------------------------------------------------------------

def train_student(
    student_cfg: SmolVLMStudentConfig,
    kd_cfg: KDConfig,
    data_cfg: DataConfig,
    paths_cfg: PathConfig,
):
    """Full SmolVLM student training pipeline with vision LoRA + projector + feature distillation."""
    print("=" * 60)
    print("  STEP 2b: Train SmolVLM Student Model (Enhanced KD)")
    print("=" * 60)
    print(f"  Student      : {student_cfg.model_path}")
    print(f"  KD Mode      : {kd_cfg.kd_mode}")
    print(f"  Vision LoRA  : {student_cfg.vision_lora}")
    print(f"  Train Proj.  : {student_cfg.train_projector}")
    print(f"  Feature KD   : {kd_cfg.feature_distillation}")
    print(f"  Feat Weight  : {kd_cfg.feature_loss_weight}")
    print(f"  Temperature  : {kd_cfg.temperature}")
    print(f"  Alpha        : {kd_cfg.alpha}")
    print(f"  Rationale    : {kd_cfg.use_rationale}")
    print(f"  Epochs       : {kd_cfg.num_epochs}")
    print("=" * 60)

    # Load teacher labels
    teacher_labels = load_jsonl(paths_cfg.teacher_labels)
    print(f"Loaded {len(teacher_labels)} teacher labels")

    # Load dataset for images
    cv_bench = load_cvbench(
        data_cfg.dataset_name, data_cfg.dataset_split,
        filter_type=data_cfg.filter_type, max_samples=data_cfg.max_samples,
    )

    # Build training data (with feature paths)
    features_dir = paths_cfg.teacher_features_dir if kd_cfg.feature_distillation else None
    training_data = build_training_data(teacher_labels, cv_bench, kd_cfg,
                                        features_dir=features_dir)
    if data_cfg.max_samples:
        training_data = training_data[:data_cfg.max_samples]
    print(f"Training samples: {len(training_data)}")

    # Check how many have features
    if kd_cfg.feature_distillation:
        n_with_feats = sum(1 for d in training_data if d.get("teacher_feat_path"))
        print(f"Samples with teacher features: {n_with_feats}/{len(training_data)}")

    # Setup model (Vision LoRA + Projector + LLM LoRA)
    model, processor = setup_student(student_cfg, kd_cfg)

    # Setup feature hooks on student vision encoder
    vision_hook_features = {}
    feature_projectors = None
    student_hooks = []

    if kd_cfg.feature_distillation:
        vision_hook_features, student_hooks = setup_student_vision_hooks(model, kd_cfg)

        if vision_hook_features is not None and student_hooks:
            # Determine feature dimensions from first available teacher feature
            feature_projectors = nn.ModuleDict()
            sample_feat_path = next(
                (d["teacher_feat_path"] for d in training_data if d.get("teacher_feat_path")),
                None
            )
            if sample_feat_path:
                sample_feats = np.load(sample_feat_path)
                for key in sample_feats.files:
                    teacher_dim = sample_feats[key].shape[-1]
                    # Infer student vision dim from model
                    base_model_ref = model.base_model.model if hasattr(model, "base_model") else model
                    vision_module, _ = _find_vision_encoder(base_model_ref)
                    if vision_module and hasattr(vision_module, "config"):
                        student_dim = getattr(vision_module.config, "hidden_size", teacher_dim)
                    else:
                        student_dim = teacher_dim
                    feature_projectors[key] = FeatureAlignmentProjector(
                        teacher_dim, student_dim, kd_cfg.feature_projector_dim
                    )
                feature_projectors = feature_projectors.to(model.device)

                # Register as attribute
                if hasattr(model, "base_model"):
                    model.base_model.feature_projectors = feature_projectors
                else:
                    model.feature_projectors = feature_projectors

                print(f"  [Feature KD] projectors created for layers: {list(feature_projectors.keys())}")

    # Collator
    collator = SmolVLMKDCollator(
        processor=processor,
        max_length=kd_cfg.max_length,
        kd_mode=kd_cfg.kd_mode,
        feature_distillation=kd_cfg.feature_distillation,
    )

    # Training args
    training_args = TrainingArguments(
        output_dir=paths_cfg.student_checkpoints,
        num_train_epochs=kd_cfg.num_epochs,
        per_device_train_batch_size=kd_cfg.batch_size,
        gradient_accumulation_steps=kd_cfg.gradient_accumulation_steps,
        learning_rate=kd_cfg.learning_rate,
        warmup_ratio=kd_cfg.warmup_ratio,
        bf16=kd_cfg.bf16,
        logging_steps=kd_cfg.logging_steps,
        save_steps=kd_cfg.save_steps,
        save_total_limit=kd_cfg.save_total_limit,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        dataloader_num_workers=0,
        report_to="none",
    )

    # Use KD Trainer
    trainer = SmolVLMKDTrainer(
        model=model,
        args=training_args,
        train_dataset=training_data,
        data_collator=collator,
        processing_class=processor.tokenizer if hasattr(processor, 'tokenizer') else processor,
        kd_cfg=kd_cfg,
        feature_projectors=feature_projectors,
        vision_hook_features=vision_hook_features,
    )

    # Train
    print("\nStarting training...")
    trainer.train()

    # Cleanup hooks
    for h in student_hooks:
        h.remove()

    # Save final checkpoint
    final_path = paths_cfg.student_final
    os.makedirs(final_path, exist_ok=True)
    trainer.save_model(final_path)
    processor.save_pretrained(final_path)
    print(f"\nStudent model saved to: {final_path}")

    return model, processor


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train SmolVLM student model with KD")
    parser.add_argument("--student_model", type=str, default="HuggingFaceTB/SmolVLM-Instruct")
    parser.add_argument("--teacher_labels", type=str,
                        default="knowledge_distillation/teacher_outputs/teacher_labels.jsonl")
    parser.add_argument("--features_dir", type=str,
                        default="knowledge_distillation/teacher_outputs/visual_features")
    parser.add_argument("--dataset", type=str, default="nyu-visionx/CV-Bench")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--kd_mode", type=str, choices=["response", "soft_label"], default="response")
    parser.add_argument("--temperature", type=float, default=3.0)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--use_rationale", action="store_true", default=True)
    parser.add_argument("--no_rationale", action="store_true")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output_dir", type=str,
                        default="knowledge_distillation/smolvlm_student_checkpoints")
    # Enhanced KD flags
    parser.add_argument("--no_vision_lora", action="store_true", help="Disable vision encoder LoRA")
    parser.add_argument("--no_projector", action="store_true", help="Keep projector frozen")
    parser.add_argument("--no_feature_kd", action="store_true", help="Disable feature distillation")
    parser.add_argument("--feature_loss_weight", type=float, default=0.3)
    parser.add_argument("--vision_lora_r", type=int, default=8)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    args = parser.parse_args()

    student_cfg = SmolVLMStudentConfig(
        model_path=args.student_model,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        vision_lora=not args.no_vision_lora,
        vision_lora_r=args.vision_lora_r,
        vision_lora_alpha=args.vision_lora_r,
        train_projector=not args.no_projector,
    )
    kd_cfg = KDConfig(
        kd_mode=args.kd_mode,
        temperature=args.temperature,
        alpha=args.alpha,
        use_rationale=args.use_rationale and not args.no_rationale,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        feature_distillation=not args.no_feature_kd,
        feature_loss_weight=args.feature_loss_weight,
    )
    data_cfg = DataConfig(
        dataset_name=args.dataset,
        dataset_split=args.split,
        max_samples=args.max_samples,
    )
    paths_cfg = PathConfig(
        teacher_labels=args.teacher_labels,
        teacher_features_dir=args.features_dir,
        student_checkpoints=args.output_dir,
        student_final=os.path.join(args.output_dir, "final"),
    )

    train_student(student_cfg, kd_cfg, data_cfg, paths_cfg)


if __name__ == "__main__":
    main()
