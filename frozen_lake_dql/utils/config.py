"""Configuration settings for the DQN agent and training."""
from dataclasses import dataclass
from typing import Optional

@dataclass
class DQNConfig:
    """Configuration for DQN agent.
    
    Attributes:
        state_size (int): Size of state space
        action_size (int): Size of action space
        hidden_size (int): Size of hidden layer in DQN
        learning_rate (float): Learning rate for optimization
        gamma (float): Discount factor
        epsilon_start (float): Initial exploration rate
        epsilon_end (float): Final exploration rate
        epsilon_decay (float): Rate of exploration decay
        memory_size (int): Size of replay memory
        batch_size (int): Size of training batch
        target_update (int): Steps between target network updates
    """
    state_size: int = 16  # 4x4 grid
    action_size: int = 4  # Left, Down, Right, Up
    hidden_size: int = 64
    learning_rate: float = 0.001
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    memory_size: int = 10000
    batch_size: int = 32
    target_update: int = 10

@dataclass
class TrainingConfig:
    """Configuration for training process.
    
    Attributes:
        num_episodes (int): Number of episodes to train
        max_steps (int): Maximum steps per episode
        render (bool): Whether to render the environment
        is_slippery (bool): Whether the environment is slippery
        log_interval (int): Episodes between logging
        save_interval (int): Episodes between saving model
        eval_interval (int): Episodes between evaluations
        num_eval_episodes (int): Number of episodes for evaluation
        model_path (str): Path to save/load model
        render_mode (Optional[str]): Render mode for gymnasium
    """
    num_episodes: int = 1000
    max_steps: int = 100
    render: bool = False
    is_slippery: bool = False
    log_interval: int = 100
    save_interval: int = 500
    eval_interval: int = 100
    num_eval_episodes: int = 10
    model_path: str = "models/saved/dqn_model.pt"
    render_mode: Optional[str] = None