"""
Centralized configuration for the Knowledge Distillation pipeline.

Teacher: Finetuned Qwen3-VL-8B (LoRA adapter: abhi26/subCV_qwen3-8B_lora)
Student: OpenGVLab/InternVL3_5-1B-Instruct
Dataset: nyu-visionx/CV-Bench (2D spatial reasoning subset)
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TeacherConfig:
    """Teacher model configuration."""
    model_path: str = "abhi26/subCV_qwen3-8B_lora"
    adapter_path: Optional[str] = None
    load_in_4bit: bool = True
    dtype: str = "bfloat16"
    generate_rationale: bool = True
    save_logits: bool = True
    save_visual_features: bool = True  # Extract intermediate vision features for feature KD
    visual_feature_layers: List[int] = field(default_factory=lambda: [-1, -4])  # Which ViT layers to extract
    rationale_max_tokens: int = 256
    rationale_temperature: float = 0.7


@dataclass
class StudentConfig:
    """Student model configuration."""
    model_path: str = "OpenGVLab/InternVL3_5-1B-Instruct"
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
        "qkv", "proj", "fc1", "fc2",
    ])
    # Projector training
    train_projector: bool = True


@dataclass
class DataConfig:
    """Dataset configuration."""
    dataset_name: str = "nyu-visionx/CV-Bench"
    dataset_split: str = "test"
    filter_type: str = "2D"  # Filter to 2D spatial tasks
    max_samples: Optional[int] = None


@dataclass
class KDConfig:
    """Knowledge distillation training configuration."""
    # KD mode
    kd_mode: str = "response"  # "response" or "soft_label"
    temperature: float = 3.0
    alpha: float = 0.7  # Weight for KD loss vs CE loss
    use_rationale: bool = True

    # Feature distillation
    feature_distillation: bool = True
    feature_loss_weight: float = 0.3  # Weight for feature MSE loss
    feature_layers: List[int] = field(default_factory=lambda: [-1, -4])  # ViT layers to match
    feature_projector_dim: int = 256  # Project teacher/student features to this dim before MSE

    # Training hyperparameters
    num_epochs: int = 1
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    vision_lr_multiplier: float = 0.1  # Lower LR for vision encoder (stabilize training)
    warmup_ratio: float = 0.1
    max_length: int = 2048
    bf16: bool = True

    # Logging & saving
    logging_steps: int = 10
    save_steps: int = 100
    save_total_limit: int = 2


@dataclass
class PathConfig:
    """Output paths."""
    base_dir: str = "knowledge_distillation"
    teacher_labels: str = "knowledge_distillation/teacher_outputs/teacher_labels.jsonl"
    teacher_features_dir: str = "knowledge_distillation/teacher_outputs/visual_features"
    student_checkpoints: str = "knowledge_distillation/student_checkpoints"
    student_final: str = "knowledge_distillation/student_checkpoints/final"
    eval_results: str = "knowledge_distillation/eval_results/student_eval.json"


@dataclass
class PipelineConfig:
    """Full pipeline configuration."""
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    student: StudentConfig = field(default_factory=StudentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    kd: KDConfig = field(default_factory=KDConfig)
    paths: PathConfig = field(default_factory=PathConfig)
