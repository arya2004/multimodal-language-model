from typing import Optional, Tuple
import torch
import torch.nn as nn

from siglip.siglip_config import SiglipConfig

class SiglipEmbeddings(nn.Module):

    def __init__(self, config: SiglipConfig ):
        super().__init__()

        self.config = config
        self.embedding_dimension = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embedding_dimension,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid"
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embedding_dimension)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False
        )

    def forward(self, pixel_value: torch.FloatTensor) -> torch.Tensor:
        #batchsize, chaannel, h,w
        _,_, heigh, width = pixel_value.shape 
        patch_embeddings = self.patch_embedding(pixel_value)
        embeddings = patch_embeddings.flatten(2)
        embeddings = embeddings.transpose(1,2)
        embeddings = embeddings + self.position_embedding(self.position_ids)

        return embeddings

