"""
Knowledge Distillation Pipeline Runner.

Orchestrates all 3 steps:
  1. Generate teacher labels (Qwen3-VL-8B finetuned → soft labels + rationales)
  2. Train student (InternVL3.5-1B on teacher outputs via LoRA)
  3. Evaluate distilled student on CV-Bench 2D test
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_step(cmd: list, step_name: str):
    """Execute a pipeline step and abort on failure."""
    print(f"\n{'=' * 60}")
    print(f"  {step_name}")
    print(f"{'=' * 60}")
    print(f"  cmd: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, cwd=str(Path(__file__).parent.parent))

    if result.returncode != 0:
        print(f"\n  ERROR: {step_name} failed (exit code {result.returncode})")
        sys.exit(1)

    print(f"\n  ✓ {step_name} done.")


def main():
    parser = argparse.ArgumentParser(description="Run the full KD pipeline")
    parser.add_argument("--skip_teacher", action="store_true",
                        help="Skip step 1 (use existing teacher labels)")
    parser.add_argument("--skip_train", action="store_true",
                        help="Skip step 2 (evaluate only)")
    parser.add_argument("--teacher_model", type=str, default="abhi26/subCV_qwen3-8B_lora")
    parser.add_argument("--adapter_path", type=str, default=None,
                        help="Separate LoRA adapter path for teacher (if not baked in)")
    parser.add_argument("--student_model", type=str, default="OpenGVLab/InternVL3_5-1B-Instruct")
    parser.add_argument("--dataset", type=str, default="nyu-visionx/CV-Bench")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--kd_mode", type=str, default="response", choices=["response", "soft_label"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_samples", type=int, default=None)
    # Enhanced KD flags
    parser.add_argument("--no_vision_lora", action="store_true", help="Disable vision encoder LoRA")
    parser.add_argument("--no_projector", action="store_true", help="Keep projector frozen")
    parser.add_argument("--no_features", action="store_true", help="Skip feature distillation entirely")
    args = parser.parse_args()

    base_dir = Path(__file__).parent
    teacher_labels = str(base_dir / "teacher_outputs" / "teacher_labels.jsonl")
    features_dir = str(base_dir / "teacher_outputs" / "visual_features")
    student_final = str(base_dir / "student_checkpoints" / "final")

    print("=" * 60)
    print("  KNOWLEDGE DISTILLATION PIPELINE")
    print("=" * 60)
    print(f"  Teacher : {args.teacher_model}")
    print(f"  Student : {args.student_model}")
    print(f"  Dataset : {args.dataset} [{args.split}] → 2D")
    print(f"  KD Mode : {args.kd_mode}")
    print(f"  Epochs  : {args.epochs}")
    print("=" * 60)

    # ----------------------------------------------------------
    # Step 1: Generate teacher labels
    # ----------------------------------------------------------
    if not args.skip_teacher:
        cmd = [
            sys.executable, str(base_dir / "generate_teacher_labels.py"),
            "--model_path", args.teacher_model,
            "--dataset", args.dataset,
            "--split", args.split,
            "--output", teacher_labels,
            "--features_dir", features_dir,
        ]
        if args.adapter_path:
            cmd += ["--adapter_path", args.adapter_path]
        if args.max_samples:
            cmd += ["--max_samples", str(args.max_samples)]
        if args.no_features:
            cmd.append("--no_features")

        run_step(cmd, "Step 1: Generate Teacher Labels")
    else:
        print("\n  [Skipped] Step 1: Using existing teacher labels.")

    # ----------------------------------------------------------
    # Step 2: Train student
    # ----------------------------------------------------------
    if not args.skip_train:
        cmd = [
            sys.executable, str(base_dir / "train_student_internvl.py"),
            "--student_model", args.student_model,
            "--teacher_labels", teacher_labels,
            "--features_dir", features_dir,
            "--dataset", args.dataset,
            "--split", args.split,
            "--kd_mode", args.kd_mode,
            "--epochs", str(args.epochs),
        ]
        if args.kd_mode == "response":
            cmd.append("--use_rationale")
        if args.max_samples:
            cmd += ["--max_samples", str(args.max_samples)]
        if args.no_vision_lora:
            cmd.append("--no_vision_lora")
        if args.no_projector:
            cmd.append("--no_projector")
        if args.no_features:
            cmd.append("--no_feature_kd")

        run_step(cmd, "Step 2: Train Student (InternVL3.5-1B)")
    else:
        print("\n  [Skipped] Step 2: Training.")

    # ----------------------------------------------------------
    # Step 3: Evaluate
    # ----------------------------------------------------------
    cmd = [
        sys.executable, str(base_dir / "evaluate_student.py"),
        "--student_path", args.student_model,
        "--adapter_path", student_final,
        "--dataset", args.dataset,
        "--split", args.split,
    ]
    if args.max_samples:
        cmd += ["--max_samples", str(args.max_samples)]

    run_step(cmd, "Step 3: Evaluate Distilled Student")

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
