from __future__ import annotations

from typing import List, Optional

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

PRETRAINED_MODEL_PATH = "shyamsn97/Mario-GPT2-700-context-length"


class MarioGPT(BaseMarioLM):
    def __init__(
        self,
        lm: Optional[PreTrainedModel] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        context_len: int = 700,
        prompter: Optional[Prompter] = None,
    ):
        super().__init__(lm, tokenizer, context_len)
        self.prompter = prompter
        if prompter is None:
            self.prompter = Prompter(self.tokenizer)

    def load_pretrained_lm(self) -> GPT2Model:
        print(f"Using {PRETRAINED_MODEL_PATH} model")
        return AutoModelWithLMHead.from_pretrained(PRETRAINED_MODEL_PATH)

    def load_pretrained_tokenizer(self) -> GPT2Tokenizer:
        print(f"Using {PRETRAINED_MODEL_PATH} tokenizer")
        return AutoTokenizer.from_pretrained(PRETRAINED_MODEL_PATH)

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
