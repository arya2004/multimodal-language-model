import torch
from torch import nn
from typing import Optional, Tuple, List
from torch.nn import CrossEntropyLoss
import math
from gemma.palligemma_config import PaliGemmaConfig
from siglip.siglip import SiglipVisionModel
from siglip.siglip_config import SiglipConfig


class PaliGemmaMultiModalProjector(nn.Module):

    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.linear = nn.Linear(config.vision_config.hidden_size, config.vision_config.projection_dim, bias=True)

    def forward(self, image_features):
        hidden_states = self.linear(image_features)
        return hidden_states