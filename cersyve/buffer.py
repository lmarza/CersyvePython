"""Replay buffer for storing counterexamples."""

import numpy as np
from typing import Tuple, Optional


class Buffer:
    """
    Replay buffer for storing state counterexamples.
    """
    def __init__(self, capacity: int, dim: int):
        """
        Args:
            capacity: Maximum number of samples to store
            dim: Dimension of each sample
        """
        self.capacity = capacity
        self.dim = dim
        self.data = np.zeros((dim, capacity), dtype=np.float32)
        self.count = np.zeros(capacity, dtype=np.int64)
        self.stored = []  # Indices of stored samples
        self.vacant = list(range(capacity))  # Indices of vacant slots
    
    def push(self, x: np.ndarray, c: Optional[np.ndarray] = None):
        """
        Add samples to the buffer.
        
        Args:
            x: Samples to add [dim, N]
            c: Count for each sample (default: zeros)
        """
        n = x.shape[1]
        assert len(self.vacant) >= n, f"Buffer overflow: trying to add {n} samples but only {len(self.vacant)} slots available"
        
        # Randomly sample vacant slots
        vacant_idx = np.random.choice(len(self.vacant), n, replace=False)
        vacant_idx = sorted(vacant_idx, reverse=True)  # Sort in reverse for popping
        
        push_idx = []
        for idx in vacant_idx:
            push_idx.append(self.vacant.pop(idx))
        
        self.stored.extend(push_idx)
        self.data[:, push_idx] = x
        
        if c is None:
            c = np.zeros(n, dtype=np.int64)
        self.count[push_idx] = c
    
    def pop(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample and remove n samples from the buffer.
        
        Args:
            n: Number of samples to pop
            
        Returns:
            x: Sampled data [dim, n]
            c: Sampled counts [n]
        """
        assert len(self.stored) >= n, f"Not enough samples: trying to pop {n} but only {len(self.stored)} available"
        
        # Randomly sample stored indices
        stored_idx = np.random.choice(len(self.stored), n, replace=False)
        stored_idx = sorted(stored_idx, reverse=True)  # Sort in reverse for popping
        
        pop_idx = []
        for idx in stored_idx:
            pop_idx.append(self.stored.pop(idx))
        
        self.vacant.extend(pop_idx)
        
        return self.data[:, pop_idx], self.count[pop_idx]
    
    def __len__(self) -> int:
        """Return number of stored samples."""
        return len(self.stored)
