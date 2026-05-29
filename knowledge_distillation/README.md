# Knowledge Distillation: Qwen3-VL-8B → InternVL3.5-1B

Transfer spatial reasoning knowledge from a finetuned **Qwen3-VL-8B** teacher to a lightweight **InternVL3.5-1B** student on the **CV-Bench 2D** test set.

| Role | Model | HF Path |
|------|-------|---------|
| Teacher | Qwen3-VL-8B (LoRA finetuned) | `abhi26/subCV_qwen3-8B_lora` |
| Student | InternVL3.5-1B-Instruct | `OpenGVLab/InternVL3_5-1B-Instruct` |
| Dataset | CV-Bench (2D spatial) | `nyu-visionx/CV-Bench` |

---

## File Structure

```
knowledge_distillation/
├── config.py                  # All dataclass configs (Teacher, Student, KD, Paths)
├── utils.py                   # Shared helpers (format_question, load_cvbench, etc.)
├── generate_teacher_labels.py # Step 1: Teacher inference → labels.jsonl
├── train_student_internvl.py  # Step 2: Train student on teacher outputs
├── evaluate_student.py        # Step 3: Evaluate distilled student
├── run_kd_pipeline.py         # Orchestrator (runs steps 1→2→3)
├── teacher_outputs/           # Generated teacher labels (auto-created)
├── student_checkpoints/       # Trained student LoRA (auto-created)
└── eval_results/              # Evaluation metrics (auto-created)
```

---

## Prerequisites

```bash
pip install unsloth datasets peft transformers accelerate bitsandbytes trl qwen-vl-utils pillow torch
```

---

## Step-by-Step Commands

### Option A: Run the full pipeline in one command

```bash
cd /path/to/VLM

python knowledge_distillation/run_kd_pipeline.py \
    --teacher_model "abhi26/subCV_qwen3-8B_lora" \
    --student_model "OpenGVLab/InternVL3_5-1B-Instruct" \
    --dataset "nyu-visionx/CV-Bench" \
    --split test \
    --kd_mode response \
    --epochs 3
```

### Option B: Run each step individually

#### Step 1 — Generate Teacher Labels

The teacher (finetuned Qwen3-VL-8B) runs inference on the 2D test split and saves hard predictions, soft-label distributions, and rationales.

```bash
python knowledge_distillation/generate_teacher_labels.py \
    --model_path "abhi26/subCV_qwen3-8B_lora" \
    --dataset "nyu-visionx/CV-Bench" \
    --split test \
    --output knowledge_distillation/teacher_outputs/teacher_labels.jsonl
```

Optional flags:
- `--max_samples 100` — limit samples for quick testing
- `--no_rationale` — skip chain-of-thought generation (faster)
- `--no_logits` — skip soft-label extraction

#### Step 2 — Train Student on Teacher Outputs

Train InternVL3.5-1B with LoRA using the teacher's responses.

```bash
# Response-based KD (recommended — trains on teacher rationales)
python knowledge_distillation/train_student_internvl.py \
    --student_model "OpenGVLab/InternVL3_5-1B-Instruct" \
    --teacher_labels knowledge_distillation/teacher_outputs/teacher_labels.jsonl \
    --dataset "nyu-visionx/CV-Bench" \
    --split test \
    --kd_mode response \
    --use_rationale \
    --epochs 3 \
    --batch_size 2 \
    --lr 2e-4
```

Alternative — soft-label KD (uses KL-divergence on probability distributions):

```bash
python knowledge_distillation/train_student_internvl.py \
    --student_model "OpenGVLab/InternVL3_5-1B-Instruct" \
    --teacher_labels knowledge_distillation/teacher_outputs/teacher_labels.jsonl \
    --dataset "nyu-visionx/CV-Bench" \
    --split test \
    --kd_mode soft_label \
    --temperature 3.0 \
    --alpha 0.7 \
    --epochs 3
```

#### Step 3 — Evaluate the Distilled Student

```bash
python knowledge_distillation/evaluate_student.py \
    --student_path "OpenGVLab/InternVL3_5-1B-Instruct" \
    --adapter_path knowledge_distillation/student_checkpoints/final \
    --dataset "nyu-visionx/CV-Bench" \
    --split test
```

---

## Quick Test (small sample)

Run the full pipeline on 20 samples to verify everything works:

```bash
python knowledge_distillation/run_kd_pipeline.py \
    --teacher_model "abhi26/subCV_qwen3-8B_lora" \
    --student_model "OpenGVLab/InternVL3_5-1B-Instruct" \
    --kd_mode response \
    --epochs 1 \
    --max_samples 20
```

---

## KD Modes Explained

| Mode | Loss | Trains On | Best For |
|------|------|-----------|----------|
| `response` | Cross-entropy | Teacher answers + rationales | Strong reasoning transfer |
| `soft_label` | α·KL-div + (1-α)·CE | Teacher probability distribution | Calibrated confidence |

---

## Outputs

After running the pipeline:

- `teacher_outputs/teacher_labels.jsonl` — one JSON per sample with prediction, soft labels, rationale
- `student_checkpoints/final/` — trained LoRA adapter (load with `PeftModel.from_pretrained`)
- `eval_results/student_eval.json` — accuracy metrics (overall + per-task)
