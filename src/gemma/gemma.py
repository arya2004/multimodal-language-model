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

class GemmaModel(nn.Module):

    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vacab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [GemmaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def get_input_embeddings(self):
        return self.embed_tokens
    
    def forward(self,attention_mask: Optional[torch.Tensor] = None, position_ids: Optional[torch.LongTensor] = None,inputs_embeds: Optional[torch.FloatTensor] = None, kv_cache: Optional[KVCache] = None) -> torch.FloatTensor:
        
        hidden_states = inputs_embeds
        normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype)
        hidden_states = hidden_states * normalizer

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(hidden_states,attention_mask=attention_mask,position_ids=position_ids,kv_cache=kv_cache)

        hidden_states = self.norm(hidden_states)
        return hidden_states


class GemmaForCausallLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = GemmaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.model.embed_tokens
    
    def tie_weights(self):
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self,attention_mask: Optional[torch.Tensor] = None, position_ids: Optional[torch.LongTensor] = None, inputs_embeds: Optional[torch.FloatTensor] = None, kv_cache: Optional[KVCache] = None) -> Tuple:
        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache,
        )

        hidden_states = outputs
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        return_data = {
            "logits": logits,
        }

        if kv_cache is not None:
            return_data["kv_cache"] = kv_cache

        return return_data