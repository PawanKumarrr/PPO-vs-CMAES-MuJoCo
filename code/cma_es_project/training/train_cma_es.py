#train_cma_es.py

import torch
import numpy as np
import yaml
import os
import random
import logging
import pandas as pd
import matplotlib.pyplot as plt

from cma_es_project.optimization.cma_es import run_cma_es
from cma_es_project.neural_network.model import PolicyNetwork
from cma_es_project.utils.logger import setup_logger
from cma_es_project.neural_network.utils import create_env

import gymnasium as gym

def main(config_path='config/cma_es.yaml', seed=42):
    logger = setup_logger()
    logger.info("Starting training using CMA-ES.")

    SEED = seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    env_name = config.get('env_name')
    # Instead of hard-coded defaults 200 / 0.0, use None so environment uses its own defaults
    max_episode_steps = config.get('max_episode_steps', None)
    reward_control_weight = config.get('reward_control_weight', None)

    config['max_episode_steps'] = max_episode_steps
    config['reward_control_weight'] = reward_control_weight

    env = create_env(env_name, config)

    # No seed here to allow full randomness in training environment resets
    env.action_space.seed(SEED)
    env.observation_space.seed(SEED)

    logger.info(f"Action Space: {env.action_space}")
    logger.info(f"Observation Space: {env.observation_space}")
    logger.info(f"Action Space Size: {env.action_space.shape}")
    logger.info(f"Observation Space Size: {env.observation_space.shape}")

    hidden_layers = config.get('hidden_layers')
    if not hidden_layers:
        raise ValueError("hidden_layers not specified in config.")

    input_size = env.observation_space.shape[0]
    output_size = env.action_space.shape[0]
    logger.info(f"Neural Network Structure: Input size={input_size}, Output size={output_size}, Hidden layers={hidden_layers}")

    policy_net = PolicyNetwork(input_size, output_size, hidden_layers)

    os.makedirs('logs/checkpoints', exist_ok=True)
    os.makedirs('logs/plots', exist_ok=True)

    best_reward, rewards_log = run_cma_es(policy_net, env, config)

    rewards_df = pd.DataFrame(rewards_log)
    rewards_df.to_csv('logs/rewards.csv', index=False)

    results = [entry['max'] for entry in rewards_log]
    np.save('logs/train_results.npy', np.array(results))

    plt.figure(figsize=(10, 6))
    plt.xlabel('Generation')
    plt.ylabel('Reward')
    plt.title(f'CMA-ES on {env_name}')
    generations = np.arange(len(results))
    results_array = np.array(results)

    window_size = 10
    if len(results_array) >= window_size:
        moving_avg = np.convolve(results_array, np.ones(window_size)/window_size, mode='valid')
        plt.plot(generations, results_array, label='Best Reward per Generation', alpha=0.6)
        plt.plot(generations[window_size-1:], moving_avg, label=f'Moving Average (window={window_size})', linewidth=2)
    else:
        plt.plot(generations, results_array, label='Best Reward per Generation')

    plt.legend()
    plt.grid(True)
    plt.savefig('logs/plots/fitness_plot.png')
    plt.close()
    logger.info("Training completed successfully.")
