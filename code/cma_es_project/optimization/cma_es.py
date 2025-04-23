#cma_es.py

import torch
import numpy as np
import random
import logging
import os
from deap import base, creator, tools, cma
from functools import partial
from multiprocessing import Pool, set_start_method
from cma_es_project.neural_network.utils import create_env

try:
    set_start_method('spawn')
except RuntimeError:
    pass

from cma_es_project.neural_network.utils import flatten_weights, update_weights
from cma_es_project.neural_network.model import PolicyNetwork
import gymnasium as gym

def parallel_eval_policy(individual, policy_net_state, input_size, output_size, hidden_layers, env_name, num_eval_episodes, action_scale, action_mean, action_low, action_high, seed, max_episode_steps, reward_control_weight):
    # Each process creates its own environment instance

    env_config = {
        'max_episode_steps': max_episode_steps,
        'reward_control_weight': reward_control_weight
    }
    env = create_env(env_name, env_config, render_mode=None)

    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Recreate model and load policy_net_state inside the worker process
    policy_net = PolicyNetwork(input_size, output_size, hidden_layers)
    policy_net.load_state_dict(policy_net_state)
    policy_net.to(device)
    policy_net.eval()

    # Update weights to this individual's parameters
    update_weights(policy_net, np.array(individual, dtype=np.float32))

    total_reward = 0.0
    for _ in range(num_eval_episodes):
        obs, info = env.reset()
        episode_reward = 0.0
        terminated = False
        truncated = False
        while not (terminated or truncated):
            obs_tensor = torch.from_numpy(obs).float().to(device)
            with torch.no_grad():
                action = policy_net(obs_tensor).cpu().numpy()
            action = action * action_scale + action_mean
            action = np.clip(action, action_low, action_high)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
        total_reward += episode_reward

    avg_reward = total_reward / num_eval_episodes
    env.close()
    return (avg_reward,)


def run_cma_es(policy_net, env, config):
    logger = logging.getLogger(__name__)

    # Log system info at start
    logger.info(f"Number of CPUs detected: {os.cpu_count()}")
    if torch.cuda.is_available():
        logger.info(f"Number of GPUs detected: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        logger.info("No GPU detected, running on CPU.")

    SEED = config.get('seed', 42)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    env_name = config.get('env_name', None)
    if env_name is None:
        raise ValueError("Environment name ('env_name') is not specified in config.")
    logger.info(f"Running CMA-ES on environment: {env_name}")

    population_size = config.get('population_size', 100)
    sigma = config.get('sigma', 0.3)
    c_m = config.get('c_m', None)
    generations = config.get('generations', 5000)
    reward_threshold = config.get('reward_threshold', 50000)
    num_eval_episodes = config.get('num_eval_episodes', 5)

    # Device setup for GPU usage if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.device_count() > 1:
        policy_net = torch.nn.DataParallel(policy_net)
    policy_net.to(device)

    # Calculate number of parameters for CMA-ES centroid
    num_params = len(flatten_weights(policy_net.module if hasattr(policy_net, 'module') else policy_net))
    if c_m is None:
        c_m = np.zeros(num_params)

    # Create Fitness and Individual if not already created
    try:
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    except RuntimeError:
        pass
    try:
        creator.create("Individual", list, fitness=creator.FitnessMax)
    except RuntimeError:
        pass

    toolbox = base.Toolbox()

    # Extract action space info
    action_space = env.action_space
    action_low = action_space.low
    action_high = action_space.high
    action_scale = (action_high - action_low) / 2.0
    action_mean = (action_high + action_low) / 2.0

    input_size = env.observation_space.shape[0]
    output_size = env.action_space.shape[0]
    hidden_layers = config.get('hidden_layers', [64, 64])

    # Flatten initial weights (not strictly necessary but can be useful)
    flat_init = flatten_weights(policy_net.module if hasattr(policy_net, 'module') else policy_net)

    # Save current policy_net weights for the worker processes
    policy_net_state = (policy_net.module if hasattr(policy_net, 'module') else policy_net).state_dict()

    # Partial function for parallel evaluation
    max_episode_steps = config.get('max_episode_steps', None)
    reward_control_weight = config.get('reward_control_weight', None)

    eval_func = partial(
        parallel_eval_policy,
        policy_net_state=policy_net_state,
        input_size=input_size,
        output_size=output_size,
        hidden_layers=hidden_layers,
        env_name=env_name,
        num_eval_episodes=num_eval_episodes,
        action_scale=action_scale,
        action_mean=action_mean,
        action_low=action_low,
        action_high=action_high,
        seed=SEED,
        max_episode_steps=max_episode_steps,
        reward_control_weight=reward_control_weight
    )


    toolbox.register("evaluate", eval_func)

    # Configure CMA-ES strategy
    strategy = cma.Strategy(centroid=c_m, sigma=sigma, lambda_=population_size)
    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)

    rewards_log = []
    best_reward = -np.inf
    best_individual = None

    # Adjust processes based on your HPC allocation (e.g., 64 CPUs)
    num_processes = 64
    logger.info(f"Using {num_processes} processes for parallel evaluations.")

    with Pool(processes=num_processes) as pool:
        for gen in range(generations):
            logger.info(f"=== Generation {gen} ===")
            population = toolbox.generate()

            # Evaluate in parallel
            fitnesses = pool.map(toolbox.evaluate, population)

            for ind, fit in zip(population, fitnesses):
                ind.fitness.values = fit

            # Update the strategy with the evaluated population
            toolbox.update(population)

            fit_values = [f[0] for f in fitnesses]
            max_fitness = max(fit_values)
            mean_fitness = np.mean(fit_values)
            std_fitness = np.std(fit_values)
            min_fitness = min(fit_values)

            logger.info(f"Max fitness: {max_fitness}, Mean: {mean_fitness}, Std: {std_fitness}, Min: {min_fitness}")

            rewards_log.append({
                'generation': gen,
                'max': max_fitness,
                'mean': mean_fitness,
                'std': std_fitness,
                'min': min_fitness
            })

            # Update best model if improved
            if max_fitness > best_reward:
                best_reward = max_fitness
                best_index = fit_values.index(max_fitness)
                best_individual = population[best_index]

                update_weights(policy_net.module if hasattr(policy_net, 'module') else policy_net, np.array(best_individual))
                os.makedirs('logs/checkpoints', exist_ok=True)
                torch.save((policy_net.module if hasattr(policy_net, 'module') else policy_net).state_dict(), 'logs/checkpoints/best_model.pth')
                logger.info("Best model updated and saved to logs/checkpoints/best_model.pth")

            # Early stopping if threshold reached
            if best_reward >= reward_threshold:
                logger.info(f"Early stopping: Reward threshold {reward_threshold} reached at generation {gen}.")
                break

    return best_reward, rewards_log
