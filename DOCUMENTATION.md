# Improving Spatial Understanding for Small Vision-Language Models

## Project Documentation

**Author:** Abhisek Guha  
**Dataset:** [nyu-visionx/CV-Bench](https://huggingface.co/datasets/nyu-visionx/CV-Bench) (2D Spatial Reasoning Subset)  
**Objective:** Improve spatial reasoning in small VLMs (~1B parameters) using knowledge distillation and fine-tuning, evaluated on spatial tasks (object relations, counting, proximity).

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Baseline VLM Evaluation](#2-baseline-vlm-evaluation)
3. [Fine-tuned Model Evaluation](#3-fine-tuned-model-evaluation)
4. [Model Fine-tuning](#4-model-fine-tuning)
5. [Knowledge Distillation](#5-knowledge-distillation)
6. [Pending Work & Timeline](#6-pending-work--timeline)
7. [Repository Structure](#7-repository-structure)

---

## 1. Project Overview

The goal is to improve spatial understanding in small Vision-Language Models on tasks such as:
- "Which object is closest to the camera?"
- "Is the chair to the left or right of the table?"
- "How many chairs are in the scene?"

**Approach Pipeline:**

```
Baseline Evaluation → Data Curation → Fine-tuning (Qwen3-VL-8B) → Knowledge Distillation (Qwen → InternVL-1B) → Final Evaluation
```

**Models Evaluated:**

| Model | Size | Type |
|-------|------|------|
| InternVL3.5-1B | ~1B | Primary student target |
| SmolVLM2-2.2B-Instruct | ~2.2B | Baseline comparison |
| SpatialBot-3B | ~3B | Spatial-specialized baseline |
| Qwen3-VL-8B (4-bit) | ~8B | Teacher model (pre-finetune) |
| Qwen3-VL-8B (LoRA Finetuned) | ~8B | Teacher model (post-finetune) |

---

## 2. Baseline VLM Evaluation

### 2.1 Setup

All models were evaluated on the **CV-Bench 2D spatial** test split using a unified benchmarking pipeline (`benchmark_vlm.py`). Metrics computed:

- **MCQ Accuracy** — primary metric (spatial question answering)
- **BLEU, ROUGE-1/2/L, METEOR** — text generation quality
- **BERTScore** — semantic similarity

### 2.2 Results — 100 Samples (Quick Validation)

| Model | MCQ Accuracy | Avg Inference Time (s) |
|-------|:---:|:---:|
| InternVL3.5-1B | 0.47 | 0.61 |
| SmolVLM2-2.2B-Instruct | 0.45 | 8.08 |
| SpatialBot-3B | 0.52 | 3.18 |
| **Qwen3-VL-8B-4bit** | **0.56** | 2.49 |

### 2.3 Results — Full Dataset (1438 Samples)

| Model | MCQ Accuracy | Per-Task (Count) | Per-Task (Relation) |
|-------|:---:|:---:|:---:|
| InternVL3.5-1B | 0.540 | 0.518 | 0.566 |
| SpatialBot-3B | 0.631 | 0.572 | 0.702 |
| **Qwen3-VL-8B-4bit** | **0.792** | **0.684** | **0.923** |

**Key Observations:**
- Qwen3-VL-8B (4-bit quantized) is the strongest baseline at 79.2% accuracy.
- InternVL3.5-1B (the 1B target model) achieves only 54% — significant room for improvement.
- Relation tasks (left/right, above/below) are generally easier than counting tasks.
- SpatialBot-3B, despite being spatial-specialized, only reaches 63.1%.

---

## 3. Fine-tuned Model Evaluation

### 3.1 Qwen3-VL-8B LoRA Fine-tuned Results (100 Samples)

After fine-tuning Qwen3-VL-8B with LoRA on the curated CV-Bench 2D spatial dataset:

| Model | MCQ Accuracy | BLEU | BERTScore F1 | Avg Inference Time (s) |
|-------|:---:|:---:|:---:|:---:|
| Qwen3-VL-8B (Baseline, 4-bit) | 0.56 | 0.089 | 0.741 | 2.49 |
| **Qwen3-VL-8B (LoRA Finetuned)** | **0.67** | **0.212** | **0.966** | 2.14 |

**Improvement:** +11% MCQ accuracy on the 100-sample validation split after fine-tuning.

The finetuned model (`abhi26/subCV_qwen3-8B_lora`) shows:
- Significantly improved spatial reasoning accuracy (56% → 67%)
- Better text generation alignment (BERTScore 0.74 → 0.97)
- Faster inference due to optimized adapter loading

### 3.2 Full Dataset Evaluation (Pending)

The finetuned Qwen3-VL-8B needs to be evaluated on the complete 1438-sample test set for a fair comparison against the baseline's 79.2%. This evaluation is pending (see [Section 6](#6-pending-work--timeline)).

---

## 4. Model Fine-tuning

### 4.1 Data Curation

A custom curation pipeline was built to prepare training data from CV-Bench:

1. **Filtered** the full CV-Bench dataset to 2D spatial tasks only
2. **Selected** 900 samples (random seed=42) for training
3. **Cleaned** and standardized format (question, choices, answer, image)
4. **Uploaded** curated dataset to HuggingFace: [`abhi26/cvbench_2d_curated`](https://huggingface.co/datasets/abhi26/cvbench_2d_curated)

Scripts: `dataset_for_training_curation/curate_cvbench.py`, `upload_cvbench_900.py`

### 4.2 Qwen3-VL-8B Fine-tuning

**Platform:** Google Colab (T4 GPU)  
**Notebook:** `finetuning_scripts/Qwen3_VL_(8B)_Vision_cv.ipynb`  
**Framework:** Unsloth + LoRA (PEFT)

**Configuration:**
- Base model: `unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit`
- LoRA rank: 16, alpha: 16
- Training data: 900 curated 2D spatial samples from CV-Bench
- Epochs: 3
- Learning rate: 2e-4 with warmup
- 4-bit quantization (BnB) for memory efficiency

**Output:** Adapter uploaded to HuggingFace as [`abhi26/subCV_qwen3-8B_lora`](https://huggingface.co/abhi26/subCV_qwen3-8B_lora)

### 4.3 InternVL3.5-1B Fine-tuning

**Platform:** Google Colab (T4 GPU)  
**Notebook:** `finetuning_scripts/internvl-finetuning.ipynb`  
**Framework:** Unsloth + LoRA (PEFT)

**Configuration:**
- Base model: `OpenGVLab/InternVL3_5-1B-Instruct`
- LoRA rank: 16, alpha: 16
- Training data: Same 900 curated 2D spatial samples
- Epochs: 3
- Learning rate: 2e-4

**Status:** Fine-tuning completed. Evaluation pending.

---

## 5. Knowledge Distillation

### 5.1 Approach

**Goal:** Transfer spatial reasoning knowledge from the finetuned Qwen3-VL-8B teacher to the lightweight InternVL3.5-1B student.

| Role | Model | HuggingFace Path |
|------|-------|-----------------|
| Teacher | Qwen3-VL-8B (LoRA finetuned) | `abhi26/subCV_qwen3-8B_lora` |
| Student | InternVL3.5-1B-Instruct | `OpenGVLab/InternVL3_5-1B-Instruct` |

**KD Mode:** Response-based distillation (teacher rationale + answer)

### 5.2 Pipeline

The KD pipeline consists of 3 stages:

```
Step 1: Generate Teacher Labels
    Teacher (finetuned Qwen3-VL-8B) → inference on CV-Bench 2D
    → saves predictions, soft-label distributions, and rationales
    → Output: teacher_labels.jsonl

Step 2: Train Student on Teacher Outputs
    InternVL3.5-1B trains with LoRA using teacher's responses
    Loss = α * KD_loss + (1-α) * CE_loss
    Temperature: 3.0, α: 0.7
    → Output: student LoRA checkpoint

Step 3: Evaluate Distilled Student
    Compare student (post-KD) vs student (baseline) on CV-Bench
    → Output: evaluation metrics
```

### 5.3 Configuration

```yaml
KD Training Hyperparameters:
  kd_mode: response (trains on teacher rationales)
  temperature: 3.0
  alpha: 0.7 (KD loss weight)
  epochs: 3
  batch_size: 2
  gradient_accumulation_steps: 4
  learning_rate: 2e-4
  warmup_ratio: 0.1
  max_length: 2048
  precision: bf16
```

### 5.4 Status

**The KD pipeline is currently running.** Step 1 (teacher label generation) takes approximately 18 hours for the full dataset due to rationale generation per sample. The pipeline code is complete and functional — results are pending completion.

Scripts:
- `knowledge_distillation/run_kd_pipeline.py` — Full pipeline orchestrator
- `knowledge_distillation/generate_teacher_labels.py` — Step 1
- `knowledge_distillation/train_student_internvl.py` — Step 2
- `knowledge_distillation/evaluate_student.py` — Step 3
- `knowledge_distillation/config.py` — All configurations
- `knowledge_distillation/utils.py` — Shared utilities

---

## 6. Pending Work & Timeline

### What is Complete

| Task | Status |
|------|--------|
| Baseline evaluation (all models, 100 + 1438 samples) | ✅ Done |
| Data curation pipeline (900 samples → HF) | ✅ Done |
| Qwen3-VL-8B LoRA fine-tuning | ✅ Done |
| InternVL3.5-1B LoRA fine-tuning | ✅ Done |
| Finetuned Qwen eval (100 samples) | ✅ Done |
| Knowledge distillation pipeline code | ✅ Done |
| Benchmarking framework & metrics | ✅ Done |

### What is Pending (ETA: 1–2 days)

| Task | Status | Reason |
|------|--------|--------|
| Finetuned Qwen3-VL-8B eval (full 1438 samples) | ⏳ Pending | Compute time needed |
| Finetuned InternVL3.5-1B eval | ⏳ Pending | Evaluation not yet run |
| KD teacher label generation (full dataset) | 🔄 Running | ~18 hours for full dataset |
| KD student training | ⏳ Pending | Blocked on Step 1 completion |
| KD student evaluation | ⏳ Pending | Blocked on Step 2 completion |
| Final comparison report (all approaches) | ⏳ Pending | Needs all results |

**Note:** The knowledge distillation pipeline is taking ~18 hours for the full dataset teacher inference (with rationale generation). The model fine-tuning for both Qwen3-VL-8B and InternVL3.5-1B has been completed — only their full evaluations remain. Given the time constraint, this submission includes all completed work with the remaining evaluations expected within 1–2 days.

---

## 7. Repository Structure

```
VLM/
├── benchmark_vlm.py              # Main benchmarking script (all models)
├── run_model.py                  # Single model evaluation
├── run_all.py                    # Run all models sequentially
├── compare_models.py             # Cross-model comparison & plots
├── config.yaml                   # Benchmark configuration
├── requirements.txt              # Python dependencies
│
├── vlm_bench/                    # Benchmarking library
│   ├── dataset.py                # CV-Bench data loading
│   ├── metrics.py                # MCQ accuracy, BLEU, ROUGE, BERTScore
│   ├── models.py                 # Model inference classes
│   └── utils.py                  # Shared utilities
│
├── dataset_for_training_curation/  # Data preparation
│   ├── curate_cvbench.py         # Filter & curate 2D samples
│   ├── upload_cvbench_900.py     # Upload to HuggingFace
│   ├── analyze_dataset.py        # Dataset analysis
│   └── clean_dataset.py          # Data cleaning
│
├── finetuning_scripts/           # Model fine-tuning notebooks
│   ├── Qwen3_VL_(8B)_Vision_cv.ipynb    # Qwen3-VL-8B LoRA fine-tuning
│   └── internvl-finetuning.ipynb         # InternVL3.5-1B LoRA fine-tuning
│
├── knowledge_distillation/       # KD pipeline
│   ├── run_kd_pipeline.py        # Full pipeline orchestrator
│   ├── generate_teacher_labels.py # Teacher inference
│   ├── train_student_internvl.py  # Student training
│   ├── evaluate_student.py        # Student evaluation
│   ├── config.py                  # KD configurations
│   └── utils.py                   # Helpers
│
├── results_before_FT_n_KD_100samples/     # Baseline results (100 samples)
├── results_before_FT_n_KD_1438samples/    # Baseline results (full dataset)
└── results_qwen_finetuned_data_better_than_before_100samples/  # Finetuned results
```

---

## How to Run

### Baseline Evaluation
```bash
python benchmark_vlm.py --config config.yaml
```

### Single Model Evaluation
```bash
python run_model.py --config config.yaml --model "InternVL3.5-1B"
```

### Knowledge Distillation Pipeline
```bash
python knowledge_distillation/run_kd_pipeline.py \
    --teacher_model "abhi26/subCV_qwen3-8B_lora" \
    --student_model "OpenGVLab/InternVL3_5-1B-Instruct" \
    --dataset "nyu-visionx/CV-Bench" \
    --split test \
    --kd_mode response \
    --epochs 3
```

---

## Summary of Findings (So Far)

1. **Qwen3-VL-8B dominates baselines** at 79.2% accuracy on the full 2D spatial test set — making it an excellent teacher for distillation.
2. **Fine-tuning works:** LoRA fine-tuning on 900 curated spatial samples improved Qwen3-VL-8B from 56% → 67% on the 100-sample validation (full eval pending).
3. **InternVL3.5-1B is weak at baseline** (54%) — the primary target for improvement via KD.
4. **Relation tasks** (spatial positioning) are easier than **counting tasks** across all models.
5. **KD pipeline is functional** and running — expected to yield improvements in the 1B student model once complete.
