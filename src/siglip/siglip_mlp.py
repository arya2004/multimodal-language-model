from typing import Optional, Tuple
import torch
import torch.nn as nn

from siglip.siglip_config import SiglipConfig



class SiglipMLP(nn.Module):
    def __init__(self, config: SiglipConfig):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # batchsize, numPatches, embedDim -> battchSize, numPatches, intermediateSize
        hidden_states = self.fc1(hidden_states)
        hidden_states = nn.functional.gelu(hidden_states, approximate="tanh")

        hidden_states = self.fc2(hidden_states)

        return hidden_states