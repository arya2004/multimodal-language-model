from typing import Optional, Tuple
import torch
import torch.nn as nn

from siglip.siglip_config import SiglipConfig
from siglip.siglip_embeddings import SiglipEmbeddings
from siglip.siglip_encoder import SiglipEncoder

class SiglipTransformer(nn.Module):

    def __init__(self, config: SiglipConfig):
        super().__init__()

        self.config = config
        embedding_dimension = config.hidden_size
        self.embeddings = SiglipEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(embedding_dimension, eps=config.layer_norm_eps)

    # converts [batchSize, channel, height, width] to [batchSize, num of patches, embedding dimension]
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        
        hidden_states = self.embeddings(pixel_values)
        last_hidden_state = self.encoder(input_embedding=hidden_states)
        last_hidden_state= self.post_layernorm(last_hidden_state)
        return last_hidden_state