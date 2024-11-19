"""Deep Q-Learning agent implementation."""
import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ..models.dqn_model import DQN
from ..utils.memory import ReplayMemory

class DQNAgent:
    """DQN Agent for interacting with environment.
    
    Attributes:
        state_size (int): Size of state space
        action_size (int): Size of action space
        device (torch.device): Device to run computations on
        policy_net (DQN): Policy network
        target_net (DQN): Target network
        memory (ReplayMemory): Replay memory buffer
        optimizer (torch.optim.Optimizer): Optimizer for policy network
    """
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_size: int = 64,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        memory_size: int = 10000,
        batch_size: int = 32,
        target_update: int = 10
    ):
        """Initialize DQN agent.
        
        Args:
            state_size (int): Size of state space
            action_size (int): Size of action space
            hidden_size (int): Size of hidden layer
            learning_rate (float): Learning rate
            gamma (float): Discount factor
            epsilon_start (float): Starting epsilon for exploration
            epsilon_end (float): Minimum epsilon
            epsilon_decay (float): Epsilon decay rate
            memory_size (int): Size of replay memory
            batch_size (int): Size of training batch
            target_update (int): Steps between target network updates
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize networks
        self.policy_net = DQN(state_size, hidden_size, action_size).to(self.device)
        self.target_net = DQN(state_size, hidden_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.memory = ReplayMemory(memory_size)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        self.steps_done = 0
    
    def select_action(self, state: int) -> int:
        """Select action using epsilon-greedy policy.
        
        Args:
            state (int): Current state
            
        Returns:
            int: Selected action
        """
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state_tensor = self._state_to_tensor(state)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()
    
    def step(self, state: int, action: int, reward: float, next_state: int, done: bool):
        """Take a step in the environment.
        
        Args:
            state (int): Current state
            action (int): Action taken
            reward (float): Reward received
            next_state (int): Next state
            done (bool): Whether episode is done
        """
        # Store experience
        self.memory.append((state, action, next_state, reward, done))
        
        # Train if enough samples
        if len(self.memory) >= self.batch_size:
            self._optimize_model()
        
        # Update target network
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.steps_done += 1
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def _optimize_model(self):
        """Perform one step of optimization on the policy network."""
        # Sample batch
        batch = self.memory.sample(self.batch_size)
        batch_states, batch_actions, batch_next_states, batch_rewards, batch_dones = zip(*batch)
        
        # Convert to tensors
        state_tensors = torch.stack([self._state_to_tensor(s) for s in batch_states])
        action_tensors = torch.tensor(batch_actions, device=self.device)
        reward_tensors = torch.tensor(batch_rewards, device=self.device)
        next_state_tensors = torch.stack([self._state_to_tensor(s) for s in batch_next_states])
        done_tensors = torch.tensor(batch_dones, device=self.device)
        
        # Compute current Q values
        current_q_values = self.policy_net(state_tensors).gather(1, action_tensors.unsqueeze(1))
        
        # Compute next Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_state_tensors).max(1)[0]
            next_q_values[done_tensors] = 0.0
            expected_q_values = reward_tensors + self.gamma * next_q_values
        
        # Compute loss and optimize
        loss = self.criterion(current_q_values, expected_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def _state_to_tensor(self, state: int) -> torch.Tensor:
        """Convert state to tensor representation.
        
        Args:
            state (int): State to convert
            
        Returns:
            torch.Tensor: One-hot encoded state tensor
        """
        tensor = torch.zeros(self.state_size, device=self.device)
        tensor[state] = 1.0
        return tensor
    
    def save(self, path: str):
        """Save agent state.
        
        Args:
            path (str): Path to save agent
        """
        self.policy_net.save(path)
    
    def load(self, path: str):
        """Load agent state.
        
        Args:
            path (str): Path to load agent from
        """
        self.policy_net.load(path)
        self.target_net.load_state_dict(self.policy_net.state_dict())