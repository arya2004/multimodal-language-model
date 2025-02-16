import torch
from torch import nn
from typing import Optional, Tuple, List
from torch.nn import CrossEntropyLoss
import math
from gemma.palligemma_config import PaliGemmaConfig
from siglip.siglip import SiglipVisionModel
from siglip.siglip_config import SiglipConfig
from gemma.gemma_config import GemmaConfig
from gemma.gemma_rms_norm import GemmaRMSNorm
from gemma.gemma_decoder_layer import GemmaDecoderLayer


class KVCache():

    def __init__(self) -> None:
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
    
    def num_items(self) -> int:
        if len(self.key_cache) == 0:
            return 0
        else:
            return self.key_cache[0].shape[-2]

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)


        return self.key_cache[layer_idx], self.value_cache[layer_idx]