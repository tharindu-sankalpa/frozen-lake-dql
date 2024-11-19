"""Visualization utilities for training metrics and agent behavior."""
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_training_metrics(
    rewards: List[float],
    steps: List[float],
    window_size: int = 100,
    save_path: Optional[str] = None
):
    """Plot training metrics.
    
    Args:
        rewards (List[float]): History of rewards
        steps (List[float]): History of steps
        window_size (int): Window size for moving average
        save_path (Optional[str]): Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    episodes = range(len(rewards))
    
    # Plot rewards
    ax1.plot(episodes, rewards, alpha=0.5, label='Raw Rewards')
    
    # Calculate and plot moving average
    if len(rewards) >= window_size:
        moving_avg = np.convolve(
            rewards,
            np.ones(window_size) / window_size,
            mode='valid'
        )
        ax1.plot(
            episodes[window_size-1:],
            moving_avg,
            label=f'Moving Average (window={window_size})'
        )
    
    ax1.set_title('Training Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.legend()
    ax1.grid(True)
    
    # Plot steps
    ax2.plot(episodes, steps, alpha=0.5, label='Steps per Episode')
    
    if len(steps) >= window_size:
        moving_avg_steps = np.convolve(
            steps,
            np.ones(window_size) / window_size,
            mode='valid'
        )
        ax2.plot(
            episodes[window_size-1:],
            moving_avg_steps,
            label=f'Moving Average (window={window_size})'
        )
    
    ax2.set_title('Steps per Episode')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_q_values(
    q_table: np.ndarray,
    save_path: Optional[str] = None
):
    """Plot Q-value heatmap.
    
    Args:
        q_table (np.ndarray): Q-value table to visualize
        save_path (Optional[str]): Path to save the plot
    """
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        q_table,
        cmap='viridis',
        xticklabels=['Left', 'Down', 'Right', 'Up'],
        cbar_kws={'label': 'Q-Value'}
    )
    
    plt.title('Q-Value Heatmap')
    plt.xlabel('Actions')
    plt.ylabel('States')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_episode_visualization(
    states: List[int],
    actions: List[int],
    rewards: List[float],
    save_path: Optional[str] = None
):
    """Visualize a single episode.
    
    Args:
        states (List[int]): States visited
        actions (List[int]): Actions taken
        rewards (List[float]): Rewards received
        save_path (Optional[str]): Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot state-action pairs
    steps = range(len(states))
    ax1.plot(steps, states, 'b.-', label='States')
    ax1_twin = ax1.twinx()
    
    # Plot actions at the same time steps
    action_steps = range(len(actions))
    ax1_twin.plot(action_steps, actions, 'r.-', label='Actions')
    
    ax1.set_title('States and Actions During Episode')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('State', color='b')
    ax1_twin.set_ylabel('Action', color='r')
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    # Plot cumulative rewards
    cumulative_rewards = np.cumsum(rewards)
    reward_steps = range(len(rewards))
    ax2.plot(reward_steps, cumulative_rewards, 'g.-', label='Cumulative Reward')
    ax2.set_title('Cumulative Reward During Episode')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Cumulative Reward')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()