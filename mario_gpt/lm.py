from typing import List, Optional

import numpy as np
import torch
from tqdm import tqdm
from transformers import (
    AutoModelWithLMHead,
    AutoTokenizer,
    GPT2Model,
    GPT2Tokenizer,
    LogitsProcessorList,
    PreTrainedModel,
    PreTrainedTokenizer,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
)

from mario_gpt.prompter import Prompter

PRETRAINED_MODEL_PATH = "shyamsn97/Mario-GPT2-700-context-length"


class MarioLM:
    def __init__(
        self,
        lm: Optional[PreTrainedModel] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        context_len: int = 700,
        prompter: Optional[Prompter] = None,
    ):
        self.context_len = context_len
        self.lm = lm

        if lm is None:
            self.lm = self.load_pretrained_lm()

        self.tokenizer = tokenizer
        if tokenizer is None:
            self.tokenizer = self.load_pretrained_tokenizer()

        self.prompter = prompter
        if prompter is None:
            self.prompter = Prompter(self.tokenizer)

    @property
    def device(self):
        return self.lm.device

    def to(self, device: torch.device):
        self.lm = self.lm.to(device)
        return self

    def load_pretrained_lm(self) -> GPT2Model:
        print(f"Using {PRETRAINED_MODEL_PATH} model")
        return AutoModelWithLMHead.from_pretrained(PRETRAINED_MODEL_PATH)

    def load_pretrained_tokenizer(self) -> GPT2Tokenizer:
        print(f"Using {PRETRAINED_MODEL_PATH} tokenizer")
        return AutoTokenizer.from_pretrained(PRETRAINED_MODEL_PATH)

    def sample_step(
        self,
        seed: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temperature: float = 2.0,
    ):
        lm = self.lm
        logits_processor = LogitsProcessorList()
        logits_warper = LogitsProcessorList(
            [
                TopKLogitsWarper(16),  # number of characters
                TemperatureLogitsWarper(temperature),
            ]
        )
        with torch.no_grad():
            attention_mask = torch.ones_like(seed).to(seed.device)
            input_ids = seed
            out = lm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                token_type_ids=None,
            )
            logits = out.logits.detach()
            if len(logits.shape) == 2:
                logits = logits.view(1, 1, -1)
            next_token_logits = logits[:, -1, :]

            next_token_scores = logits_processor(input_ids, next_token_logits)
            next_token_scores = logits_warper(input_ids, next_token_scores)
            probs = torch.nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        return next_tokens, encoder_hidden_states

    def sample(
        self,
        seed: Optional[torch.Tensor] = None,
        prompts: Optional[List[str]] = None,
        num_steps: int = 1,
        temperature: float = 2.0,
        encoder_hidden_states: torch.Tensor = None,
        use_tqdm: bool = False,
    ):
        context_len = self.context_len - 28
        self.lm.eval()
        with torch.no_grad():
            if seed is None:
                seed = self.tokenizer("X", return_tensors="pt").input_ids.view(1, 1)
            out = seed.to(self.device)
            if encoder_hidden_states is None:
                if prompts is not None:
                    encoder_hidden_states = torch.stack(
                        [self.prompter.output_hidden(prompt) for prompt in prompts]
                    )
                else:
                    encoder_hidden_states = torch.stack(
                        [
                            self.prompter(sample_prompt=True)[1]
                            for _ in range(seed.shape[0])
                        ]
                    )
            encoder_hidden_states = encoder_hidden_states.to(
                self.device
            )  # b x 1 x hidden_dim
            encoder_hidden_states = encoder_hidden_states.view(seed.shape[0], 1, -1)
            if not use_tqdm:
                bar = np.arange(num_steps)
            else:
                bar = tqdm(np.arange(num_steps))
            with torch.no_grad():
                for i in bar:
                    inp = out * 1
                    if len(out.shape) > 0 and out.shape[-1] > context_len:
                        diff = inp.shape[-1] % 14  # height of mario level
                        ctx = context_len + diff
                        inp = inp[:, -ctx:] * 1
                    next_tokens, encoder_hidden_states = self.sample_step(
                        inp,
                        encoder_hidden_states=encoder_hidden_states,
                        temperature=temperature,
                    )
                    out = torch.cat([out, next_tokens.unsqueeze(-1)], dim=-1)
                    if use_tqdm:
                        bar.set_description(
                            f"shape: {inp.shape}, {out.shape} first: {inp[0][0]}, last: {out[0][-1]}"
                        )
            if use_tqdm:
                bar.close()
        self.lm.train()
        return out
