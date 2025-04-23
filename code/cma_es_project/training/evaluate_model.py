#evaluate_model.py

import os
import torch
import numpy as np
import yaml
import logging
import imageio
import gymnasium as gym

from cma_es_project.neural_network.model import PolicyNetwork
from cma_es_project.neural_network.utils import load_model
from cma_es_project.utils.logger import setup_logger
from cma_es_project.neural_network.utils import create_env


def main(checkpoint_path='logs/checkpoints/best_model.pth', config_path='config/cma_es.yaml', seed=42):
    logger = setup_logger()
    logger.info("Starting model evaluation.")

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["MUJOCO_GL"] = "egl"

    SEED = seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
    except Exception as e:
        logger.error(f"Failed to load configuration file {config_path}: {e}")
        return

    env_name = config.get('env_name')
    hidden_layers = config.get('hidden_layers')
    num_eval_episodes = config.get('num_eval_episodes', 5)

    if not env_name:
        logger.error("Environment name ('env_name') not specified.")
        return

    if not hidden_layers:
        logger.error("Hidden layers not specified in config file.")
        return


    max_episode_steps = config.get('max_episode_steps', None)
    reward_control_weight = config.get('reward_control_weight', None)

    config['max_episode_steps'] = max_episode_steps
    config['reward_control_weight'] = reward_control_weight

    env = create_env(env_name, config, render_mode='rgb_array')

    
    # No fixed seed in evaluation reset to ensure random target positions
    env.action_space.seed(SEED)
    env.observation_space.seed(SEED)

    logger.info(f"Action Space: {env.action_space}")
    logger.info(f"Observation Space: {env.observation_space}")
    logger.info(f"Action Space Size: {env.action_space.shape}")
    logger.info(f"Observation Space Size: {env.observation_space.shape}")

    input_size = env.observation_space.shape[0]
    output_size = env.action_space.shape[0]
    logger.info(f"Neural Network Structure: Input size={input_size}, Output size={output_size}, Hidden layers={hidden_layers}")

    device = torch.device('cpu')
    model = PolicyNetwork(input_size, output_size, hidden_layers)

    try:
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        logger.info(f"Model loaded successfully from {checkpoint_path}")

        print("Loaded model parameters:")
        for param_tensor in model.state_dict():
            print(f"{param_tensor}: {model.state_dict()[param_tensor].size()}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return

    action_low = env.action_space.low
    action_high = env.action_space.high
    action_scale = (action_high - action_low) / 2.0
    action_mean = (action_high + action_low) / 2.0

    frames = []
    total_rewards = []
    steps_list = []

    for episode in range(num_eval_episodes):
        obs, info = env.reset()
        episode_reward = 0.0
        steps = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            frame = env.render()
            if frame is not None:
                frames.append(frame)

            obs_tensor = torch.from_numpy(obs).float().to(device)
            with torch.no_grad():
                action = model(obs_tensor).cpu().numpy()

            action = action * action_scale + action_mean
            action = np.clip(action, action_low, action_high)

            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1

        total_rewards.append(episode_reward)
        steps_list.append(steps)
        logger.info(f"Episode {episode+1}: Total Reward: {episode_reward}, Steps Taken: {steps}")

    avg_reward = np.mean(total_rewards)
    avg_steps = np.mean(steps_list)
    logger.info(f"Average Reward over {num_eval_episodes} episodes: {avg_reward}, Average Steps Taken: {avg_steps}")
    print(f"Average Reward over {num_eval_episodes}: {avg_reward}, Steps Taken: {avg_steps}")

    video_path = 'logs/videos/manual_video.mp4'
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    imageio.mimsave(video_path, frames, fps=30)
    logger.info(f"Video saved to {video_path}")

    env.close()
