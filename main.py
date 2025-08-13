"""
Multi-Modal Open World Continual Learning (MMOWCL) 

Usage:
    python main.py -d mmea -i 0
    python main.py -d mmea -i 1
"""

import argparse
import json
import os
import socket
import datetime
import uuid
import logging
import warnings
from trainer.trainer import train
from utils.utils import shallow_merge

import wandb


# Suppress unnecessary warnings for cleaner output
warnings.filterwarnings("ignore") 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, required=True)
    parser.add_argument('-m', '--model_name', type=str, required=True)
    parser.add_argument('--wandb_project', type=str, default='MMEA-OWCL')
    parser.add_argument('--wandb_entity', type=str, default='mmea-owcl')
    parser.add_argument('--debug_mode', action='store_true', help='Enable debug mode with reduced steps and no W&B logging')
    args, _ = parser.parse_known_args()

    # Load config file from args/{dataset}/exp_{id}.json
    # config_path = os.path.join("exps", args.dataset, f"exp_{args.model_name}.json")
    config_path = os.path.join("exps", f"exp_{args.model_name}.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)

    args_dict = vars(args)           # argparse.Namespace → dict 변환
    config.update(args_dict)         # argparse 값이 JSON 값을 덮어씀

    # init_cls
    init_cls = config.get('init_cls', None)
    if init_cls is None:
        config['init_cls'] = config['increment']

    # Add unique_id, timestamp and host to the config
    config['run_id'] = str(uuid.uuid4()).split('-')[0]
    config['timestamp'] = str(datetime.datetime.now())
    config['host'] = socket.gethostname()
    config['use_wandb'] = bool(config['wandb_project'] and config['wandb_entity'] )

    if config['debug_mode']:
        print('Debug mode enabled: running only a few forward steps per epoch with W&B disabled.')
        config['use_wandb'] = 0

    if config['use_wandb']:
        # Init wandb
        wandb.init(
            project=config['wandb_project'],
            entity=config['wandb_entity'],
            name=f"{config['model_name']}_{config['run_id']}",
            config=config
        )
        sweep_cfg = dict(wandb.config)  # W&B가 정리한 최종 설정
        config.update(sweep_cfg)        # JSON < argparse < W&B(sweep)
    
    # Print experiment summary
    print("=" * 60)
    print("🚀 Multi-Modal Open World Continual Learning")
    print("=" * 60)
    print(f"✓ Dataset: {config['dataset']}")
    print(f"✓ Model: {config['model_name']}")
    print(f"✓ Modalities: {config['modality']}")
    print(f"✓ Tasks: Initial {config['init_cls']} classes + {config['increment']} classes each increment")
    print(f"✓ OOD Methods: {config['ood_methods']}")
    print("=" * 60)

    # Start training
    train(config)


if __name__ == '__main__':
    main()