
#plotter.py


import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def plot_fitness_progression(log_file='logs/rewards.csv', save_path='logs/plots/fitness_plot.png'):
    """
    Plots the fitness progression over generations and saves the plot as an image.

    Args:
        log_file (str): Path to the CSV file containing rewards log.
        save_path (str): Path to save the generated plot.
    """
    if not os.path.exists(log_file):
        print(f"Log file not found: {log_file}")
        return

    # Load the rewards log
    rewards_log = pd.read_csv(log_file)
    
    generations = rewards_log['generation']
    max_rewards = rewards_log['max']
    mean_rewards = rewards_log['mean']
    std_rewards = rewards_log['std']

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(generations, max_rewards, label='Max Reward')
    plt.plot(generations, mean_rewards, label='Mean Reward')
    plt.fill_between(generations, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.2, label='Mean ± Std')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Fitness Progression')
    plt.legend()
    plt.grid(True)

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Fitness plot saved to {save_path}")
