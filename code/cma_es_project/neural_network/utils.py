#utils.py
import torch
import os
import numpy as np
import gymnasium as gym

def create_env(env_name, config, render_mode=None):
    env_kwargs = {}

    if 'max_episode_steps' in config and config['max_episode_steps'] is not None:
        env_kwargs['max_episode_steps'] = config['max_episode_steps']
        
    # New code (correct):
    if 'reward_control_weight' in config and config['reward_control_weight'] is not None:
        env_kwargs['reward_control_weight'] = config['reward_control_weight']
    
    if render_mode is not None:
        env_kwargs['render_mode'] = render_mode

    env = gym.make(env_name, **env_kwargs)
    return env


def save_model(model, filepath):
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(model.state_dict(), filepath)
        print(f"Model saved successfully to {filepath}")
    except Exception as e:
        print(f"Error saving model: {e}")

def load_model(model, filepath, device='cpu'):
    try:
        # Use weights_only=True once PyTorch supports it by default
        state_dict = torch.load(filepath, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        print(f"Model loaded successfully from {filepath}")
    except Exception as e:
        print(f"Error loading model: {e}")
    return model

def flatten_weights(model):
    params = []
    for param in model.parameters():
        params.append(param.detach().cpu().numpy().flatten())
    flat_params = np.concatenate(params)
    return flat_params

def update_weights(model, flat_params):
    idx = 0
    for param in model.parameters():
        param_length = param.numel()
        param_shape = param.shape
        param_data = flat_params[idx:idx+param_length].astype(np.float32)
        param.data = torch.from_numpy(param_data.reshape(param_shape)).to(param.device)
        idx += param_length
