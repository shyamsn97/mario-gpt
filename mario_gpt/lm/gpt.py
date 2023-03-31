from __future__ import annotations
import sys
from typing import Any, Dict, List, Optional

import torch
from transformers import (
    AutoModelWithLMHead,
    AutoTokenizer,
    GPT2Model,
    GPT2Tokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from mario_gpt.lm.base import BaseMarioLM
from mario_gpt.prompter import Prompter
from mario_gpt.sampler import GPTSampler, SampleOutput

# append project directory to path so predict.py can be imported
sys.path.append('.')

from predict import MODEL_CACHE, MODEL_ID


class MarioGPT(BaseMarioLM):
    PRETRAINED_LM_PATH = MODEL_ID
    PRETRAINED_TOKENIZER_PATH = MODEL_ID

    BASE_LM_PATH = "distilgpt2"
    BASE_TOKENIZER_PATH = "distilgpt2"

    def __init__(
        self,
        lm: Optional[PreTrainedModel] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        context_len: int = 700,
        prompter: Optional[Prompter] = None,
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
        self.prompter = prompter
        if prompter is None:
            self.prompter = Prompter(self.tokenizer)

    def generate_seed(self, length: int, batch_size: Optional[int] = None):
        seed = self.tokenizer("X", return_tensors="pt").input_ids.squeeze()
        if batch_size is None:
            return seed.repeat(length)
        return seed.view(1, 1).repeat(batch_size, length)

    def load_pretrained_lm(self, path: str, lm_kwargs: Dict[str, Any]) -> GPT2Model:
        return AutoModelWithLMHead.from_pretrained(
            path, **{**lm_kwargs, "add_cross_attention": True, "cache_dir": MODEL_CACHE,
                     "local_files_only": True, }
        )

    def load_pretrained_tokenizer(
        self, path: str, tokenizer_kwargs: Dict[str, Any]
    ) -> GPT2Tokenizer:
        return AutoTokenizer.from_pretrained(path, **{**tokenizer_kwargs, "add_cross_attention": True, "cache_dir": MODEL_CACHE,
                     "local_files_only": True, })

    def sample(
        self,
        seed: Optional[torch.Tensor] = None,
        prompts: Optional[List[str]] = None,
        num_steps: int = 1,
        temperature: float = 2.0,
        encoder_hidden_states: torch.Tensor = None,
        use_tqdm: bool = False,
        return_tensor: bool = False,
    ) -> SampleOutput:
        sampler = GPTSampler(self, temperature, 16, self.context_len, use_tqdm)
        return sampler(
            seed=seed,
            prompts=prompts,
            num_steps=num_steps,
            encoder_hidden_states=encoder_hidden_states,
            return_tensor=return_tensor,
        )
