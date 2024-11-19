"""Deep Q-Network model implementation."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    """Deep Q-Network architecture.
    
    Attributes:
        fc1 (nn.Linear): First fully connected layer
        out (nn.Linear): Output layer
    """
    
    def __init__(self, in_states: int, hidden_size: int, out_actions: int):
        """Initialize DQN.
        
        Args:
            in_states (int): Number of input states
            hidden_size (int): Number of hidden layer nodes
            out_actions (int): Number of possible actions
        """
        super().__init__()
        self.fc1 = nn.Linear(in_states, hidden_size)
        self.out = nn.Linear(hidden_size, out_actions)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Q-values for each action
        """
        x = F.relu(self.fc1(x))
        return self.out(x)
    
    def save(self, path: str):
        """Save model state to file.
        
        Args:
            path (str): Path to save model
        """
        torch.save(self.state_dict(), path)
    
    def load(self, path: str):
        """Load model state from file.
        
        Args:
            path (str): Path to load model from
        """
        self.load_state_dict(torch.load(path))