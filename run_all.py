"""
Run all models sequentially then compare.

Usage:
    python run_all.py --config config.yaml
"""

import argparse
import yaml
import torch

from run_model import evaluate_model
from compare_models import main as compare_main


def main():
    parser = argparse.ArgumentParser(description="Run all VLM models and compare results")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config YAML file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    print("\n" + "="*60)
    print("VLM SPATIAL UNDERSTANDING BENCHMARK — FULL RUN")
    print("="*60)
    print(f"  Config: {args.config}")
    print(f"  Models: {[m['name'] for m in config['models']]}")
    print(f"  Dataset: {config['dataset']['name']}")
    print(f"  Samples: {config['dataset']['num_samples']}")
    print(f"  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # Evaluate each model
    for model_config in config['models']:
        evaluate_model(config, model_config)

    # Run comparison
    print("\n\n" + "="*60)
    print("RUNNING COMPARISON...")
    print("="*60)

    # Reuse compare_models logic via CLI simulation
    import sys
    sys.argv = ['compare_models.py', '--config', args.config]
    compare_main()


if __name__ == "__main__":
    main()
