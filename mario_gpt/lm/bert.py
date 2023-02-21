from __future__ import annotations

from typing import Optional

from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    RobertaModel,
    RobertaTokenizer,
)

from mario_gpt.lm.base import BaseMarioLM

PRETRAINED_MODEL_MASK_PATH = "shyamsn97/MarioBert-448-inpaint-context-length"


class MarioBert(BaseMarioLM):
    def __init__(
        self,
        lm: Optional[PreTrainedModel] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        context_len: int = 700,
        mask_proportion: float = 0.15,
    ):
        super().__init__(lm, tokenizer, context_len)
        self.mask_proportion = mask_proportion

    def generate_seed(self, length: int, batch_size: Optional[int] = None):
        seed = self.tokenizer("X", return_tensors="pt").input_ids.squeeze()[
            1:-1
        ]  # remove start and end tokens
        if batch_size is None:
            return seed.repeat(length)
        return seed.view(1, 1).repeat(batch_size, length)

    def load_pretrained_lm(self) -> RobertaModel:
        print(f"Using {PRETRAINED_MODEL_MASK_PATH} model")
        return AutoModelForMaskedLM.from_pretrained(PRETRAINED_MODEL_MASK_PATH)

    def load_pretrained_tokenizer(self) -> RobertaTokenizer:
        print(f"Using {PRETRAINED_MODEL_MASK_PATH} tokenizer")
        return AutoTokenizer.from_pretrained(PRETRAINED_MODEL_MASK_PATH)
