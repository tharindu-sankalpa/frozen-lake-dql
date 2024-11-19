# frozen_lake_dql/utils/memory.py
"""Experience replay memory implementation."""
from collections import deque
import random
from typing import List, Tuple

import torch

class ReplayMemory:
    """Replay memory for experience replay in DQN.
    
    Attributes:
        memory (deque): Circular buffer to store experiences
        max_size (int): Maximum size of memory buffer
    """
    
    def __init__(self, max_size: int):
        """Initialize replay memory.
        
        Args:
            max_size (int): Maximum size of memory buffer
        """
        self.memory = deque(maxlen=max_size)
        self.max_size = max_size
    
    def append(self, experience: Tuple[int, int, int, float, bool]):
        """Add experience to memory.
        
        Args:
            experience (tuple): (state, action, next_state, reward, done)
        """
        self.memory.append(experience)
    
    def sample(self, batch_size: int) -> List[Tuple]:
        """Sample random batch from memory.
        
        Args:
            batch_size (int): Size of batch to sample
            
        Returns:
            List[Tuple]: Batch of experiences
        """
        return random.sample(self.memory, batch_size)
    
    def __len__(self) -> int:
        """Get current size of memory.
        
        Returns:
            int: Current size of memory
        """
        return len(self.memory)