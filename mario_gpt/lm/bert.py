from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    RobertaModel,
    RobertaTokenizer,
)

from mario_gpt.lm.base import BaseMarioLM

PRETRAINED_MODEL_PATH = "shyamsn97/MarioBert-448-inpaint-context-length"


class MarioBert(BaseMarioLM):
    PRETRAINED_LM_PATH = PRETRAINED_MODEL_PATH
    PRETRAINED_TOKENIZER_PATH = PRETRAINED_MODEL_PATH

    BASE_LM_PATH = "distilroberta-base"
    BASE_TOKENIZER_PATH = "distilroberta-base"

    def __init__(
        self,
        lm: Optional[PreTrainedModel] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        context_len: int = 448,
        mask_proportion: float = 0.16,
        lm_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        lm_kwargs: Dict[str, Any] = {},
        tokenizer_kwargs: Dict[str, Any] = {},
    ):
        super().__init__(
            lm,
            tokenizer,
            context_len,
            lm_path,
            tokenizer_path,
            lm_kwargs,
            tokenizer_kwargs,
        )
        self.mask_proportion = mask_proportion
        self.mask_portion = int(self.context_len * self.mask_proportion)

    def sample_mask(self, input_ids):
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[-1]
        mask_portion = self.mask_portion
        sampled_start_idx = [i for i in range(seq_len - mask_portion) if i % 14 == 0]
        sampled_start_idx = np.random.choice(sampled_start_idx, batch_size)
        sampled_masks = []
        for idx in sampled_start_idx:
            mask = torch.arange(idx, idx + mask_portion)
            sampled_masks.append(mask)
        sampled_mask_indices = torch.stack(sampled_masks)
        return self.apply_mask(input_ids, sampled_mask_indices)

    def generate_mask(self, mask_len: int, batch_size: int = 1):
        mask_token = self.tokenizer("<mask>").input_ids[1]
        ones = torch.ones((batch_size, mask_len))
        return ones * mask_token

    def apply_mask(self, level, masked_indices, mask=None):
        if len(level.shape) == 1:
            level = level.unsqueeze(0)
        batch_size = level.shape[0]
        mask_len = masked_indices.shape[-1]
        if mask is None:
            mask = self.generate_mask(mask_len, batch_size)
        mask = mask.long().to(level.device)
        masked_level = level * torch.ones_like(level).to(level.device)
        masked_level[:, masked_indices] = mask
        return masked_level

    def generate_seed(self, length: int, batch_size: Optional[int] = None):
        seed = self.tokenizer("X", return_tensors="pt").input_ids.squeeze()[
            1:-1
        ]  # remove start and end tokens
        if batch_size is None:
            return seed.repeat(length)
        return seed.view(1, 1).repeat(batch_size, length)

    def load_pretrained_lm(self, path: str, lm_kwargs: Dict[str, Any]) -> RobertaModel:
        return AutoModelForMaskedLM.from_pretrained(path, **lm_kwargs)

    def load_pretrained_tokenizer(
        self, path: str, tokenizer_kwargs: Dict[str, Any]
    ) -> RobertaTokenizer:
        return AutoTokenizer.from_pretrained(path, **tokenizer_kwargs)
