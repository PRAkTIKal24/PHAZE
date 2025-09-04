import torch
import torch.nn as nn

class SimpleEarlyExitModel(nn.Module):
    """A simple toy model to demonstrate early-exit capabilities."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(20, 5) # Early-exit layer

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)


