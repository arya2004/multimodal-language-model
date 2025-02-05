from typing import Optional, Tuple
import torch
import torch.nn as nn

from siglip.siglip_config import SiglipConfig
from siglip.siglip_mlp import SiglipMLP
from siglip.siglip_attention import SiglipAttention

class SiglipEncoderLayer(nn.Module):
    def __init__(self, config: SiglipConfig):
        super().__init__()
        self.embedding_dimension = config.hidden_size
        self.self_attention = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embedding_dimension, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embedding_dimension, eps=config.layer_norm_eps)

    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # rsidual: batchsize, numPatches, embedDim
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, _ = self.self_attention(hidden_states=hidden_states)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        # mlp has no mizing of patches, adds more params so more degree of freedom. Adds non lineriaty
        hidden_states = self.mlp(hidden_states) 
        #skip connection
        hidden_states = residual + hidden_states

        return hidden_states