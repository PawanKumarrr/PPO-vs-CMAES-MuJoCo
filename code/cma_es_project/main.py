#main.py

import os
import sys
import logging
import argparse
import random
import numpy as np
import torch

from cma_es_project.training import train_cma_es
from cma_es_project.training import evaluate_model

def set_global_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logging.info(f"Random seed set to {seed}")

def validate_file(filepath, file_description):
    if not os.path.isfile(filepath):
        logging.error(f"{file_description} not found at: {filepath}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='CMA-ES Optimization Project Entry Point')
    parser.add_argument('--mode', choices=['train', 'evaluate'], required=True,
                        help='Mode to run the script: train or evaluate.')
    parser.add_argument('--config', type=str, default='config/cma_es.yaml',
                        help='Path to the configuration file (default: config/cma_es.yaml).')
    parser.add_argument('--checkpoint', type=str, default='logs/checkpoints/best_model.pth',
                        help='Path to the model checkpoint for evaluation (default: logs/checkpoints/best_model.pth).')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42).')

    args = parser.parse_args()

    set_global_seeds(args.seed)

    if args.mode == 'train':
        validate_file(args.config, 'Configuration file')
        try:
            logging.info("Starting training process...")
            train_cma_es.main(config_path=args.config, seed=args.seed)
            logging.info("Training completed successfully.")
        except Exception as e:
            logging.error(f"An error occurred during training: {e}")
            sys.exit(1)
    elif args.mode == 'evaluate':
        validate_file(args.checkpoint, 'Checkpoint file')
        try:
            logging.info("Starting evaluation process...")
            evaluate_model.main(checkpoint_path=args.checkpoint, config_path=args.config, seed=args.seed)
            logging.info("Evaluation completed successfully.")
        except Exception as e:
            logging.error(f"An error occurred during evaluation: {e}")
            sys.exit(1)
    else:
        logging.error(f"Invalid mode selected: {args.mode}")
        sys.exit(1)

if __name__ == "__main__":
    main()
