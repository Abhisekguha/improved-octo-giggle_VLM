"""Dataset loading utilities for VLM benchmarking."""

import pandas as pd
from datasets import load_dataset


def load_cvbench_data(config):
    """Load CV-Bench dataset and filter by type."""
    print("\n" + "="*60)
    print("LOADING CV-BENCH DATASET")
    print("="*60)

    dataset = load_dataset(config['dataset']['name'], split=config['dataset']['split'])

    # Filter by type (2d/3d)
    type_filter = config['dataset'].get('type_filter', 'all')
    if type_filter != 'all':
        dataset = dataset.filter(lambda x: x['type'].lower() == type_filter.lower())
        print(f"  Filtered to type='{type_filter}': {len(dataset)} samples")

    # Take first N samples
    num_samples = config['dataset']['num_samples']
    if num_samples < len(dataset):
        dataset = dataset.select(range(num_samples))

    print(f"  Using {len(dataset)} samples for evaluation")

    # Print task distribution
    tasks = [s['task'] for s in dataset]
    task_counts = pd.Series(tasks).value_counts()
    print(f"\n  Task distribution:")
    for task, count in task_counts.items():
        print(f"    {task}: {count}")

    return dataset
