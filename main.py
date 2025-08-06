"""
Multi-Modal Open World Continual Learning (MMOWCL) 

Usage:
    python main.py -d mmea -i 0
    python main.py -d mmea -i 1

    python main.py --dataset mydataset --id 1
"""

import argparse
import json
import os
import warnings
from trainer.trainer import train

# Suppress unnecessary warnings for cleaner output
warnings.filterwarnings("ignore") 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, required=True)
    parser.add_argument('-i', '--id', type=int, required=True)
    args = parser.parse_args()

    # Load config file from args/{dataset}/exp_{id}.json
    config_path = os.path.join("args", args.dataset, f"exp_{args.id}.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    # Print experiment summary
    print("=" * 60)
    print("🚀 Multi-Modal Open World Continual Learning")
    print("=" * 60)
    print(f"✓ Dataset: {args.dataset}")
    print(f"✓ Experiment ID: {args.id}")
    print(f"✓ Model: {config['model_name']}")
    print(f"✓ Modalities: {config['modality']}")
    print(f"✓ Tasks: Initial {config['init_cls']} classes + {config['increment']} classes each increment")
    print(f"✓ OOD Methods: {config['ood_methods']}")
    print("=" * 60)

    # Start training
    train(config)

if __name__ == '__main__':
    main()