from typing import Optional, Tuple
import torch
import torch.nn as nn

from siglip.siglip_config import SiglipConfig
from siglip.siglip_transformer import SiglipTransformer

class SiglipVisionModel(nn.Module):

    def __init__(self, config: SiglipConfig):
        super().__init__()
        self.config = config  
        self.vision_model = SiglipTransformer(config)

    # converts [batchSize, channel, height, width] to [batchSize, num of patches, embedding dimension]
    def forward(self, pixel_value) -> Tuple:
        return self.vision_model(pixel_value=pixel_value)