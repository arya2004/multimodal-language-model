import torch
from torch import nn
from typing import Optional, Tuple, List
from torch.nn import CrossEntropyLoss
import math
from gemma.palligemma_config import PaliGemmaConfig
from siglip.siglip import SiglipVisionModel
from siglip.siglip_config import SiglipConfig
from gemma.palligemma_multimodal_projector import PaliGemmaMultiModalProjector

class GemmaRMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x):
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)
    