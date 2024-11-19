# frozen_lake_dql/training/trainer.py
"""Trainer for DQN agent in FrozenLake environment."""
import time
from typing import List, Tuple

import gymnasium as gym
import numpy as np
from tqdm import tqdm

from ..agents.dqn_agent import DQNAgent
from ..utils.config import DQNConfig, TrainingConfig


class FrozenLakeTrainer:
    """Trainer for DQN agent in FrozenLake environment.
    
    Attributes:
        agent (DQNAgent): The DQN agent
        env (gym.Env): The FrozenLake environment
        config (TrainingConfig): Training configuration
        rewards_history (List[float]): History of rewards
        steps_history (List[int]): History of steps per episode
    """
    
    def __init__(
        self,
        agent_config: DQNConfig = DQNConfig(),
        training_config: TrainingConfig = TrainingConfig()
    ):
        """Initialize trainer.
        
        Args:
            agent_config (DQNConfig): Agent configuration
            training_config (TrainingConfig): Training configuration
        """
        self.config = training_config
        self.env = gym.make(
            "FrozenLake-v1",
            map_name="4x4",
            is_slippery=training_config.is_slippery,
            render_mode=training_config.render_mode
        )
        
        self.agent = DQNAgent(
            state_size=agent_config.state_size,
            action_size=agent_config.action_size,
            hidden_size=agent_config.hidden_size,
            learning_rate=agent_config.learning_rate,
            gamma=agent_config.gamma,
            epsilon_start=agent_config.epsilon_start,
            epsilon_end=agent_config.epsilon_end,
            epsilon_decay=agent_config.epsilon_decay,
            memory_size=agent_config.memory_size,
            batch_size=agent_config.batch_size,
            target_update=agent_config.target_update
        )
        
        self.rewards_history = []
        self.steps_history = []
    
    def train(self) -> Tuple[List[float], List[int]]:
        """Train the agent.
        
        Returns:
            Tuple[List[float], List[int]]: Rewards and steps history
        """
        progress_bar = tqdm(range(self.config.num_episodes), desc="Training")
        
        for episode in progress_bar:
            episode_reward, episode_steps = self._train_episode()
            
            self.rewards_history.append(episode_reward)
            self.steps_history.append(episode_steps)
            
            # Update progress bar
            avg_reward = np.mean(self.rewards_history[-100:])
            progress_bar.set_postfix({
                'reward': f'{episode_reward:.2f}',
                'avg_reward': f'{avg_reward:.2f}',
                'epsilon': f'{self.agent.epsilon:.2f}'
            })
            
            # Periodic evaluation
            if episode > 0 and episode % self.config.eval_interval == 0:
                self._evaluate()
            
            # Save model
            if episode > 0 and episode % self.config.save_interval == 0:
                self.agent.save(self.config.model_path)
        
        return self.rewards_history, self.steps_history
    
    def _train_episode(self) -> Tuple[float, int]:
        """Train for one episode.
        
        Returns:
            Tuple[float, int]: Episode reward and steps
        """
        state, _ = self.env.reset()
        episode_reward = 0
        steps = 0
        
        for step in range(self.config.max_steps):
            # Select and perform action
            action = self.agent.select_action(state)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            
            # Store transition and optimize
            self.agent.step(state, action, reward, next_state, terminated)
            
            episode_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
                
            state = next_state
        
        return episode_reward, steps
    
    def _evaluate(self, render: bool = False) -> float:
        """Evaluate the agent's performance.
        
        Args:
            render (bool): Whether to render the evaluation
            
        Returns:
            float: Average reward over evaluation episodes
        """
        eval_rewards = []
        
        for _ in range(self.config.num_eval_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            
            for _ in range(self.config.max_steps):
                if render:
                    self.env.render()
                    time.sleep(0.1)
                
                action = self.agent.select_action(state)
                state, reward, terminated, truncated, _ = self.env.step(action)
                episode_reward += reward
                
                if terminated or truncated:
                    break
            
            eval_rewards.append(episode_reward)
        
        avg_reward = np.mean(eval_rewards)
        print(f"\nEvaluation Average Reward: {avg_reward:.2f}")
        return avg_reward
    
    def close(self):
        """Close the environment."""
        self.env.close()