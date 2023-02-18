from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from PIL.Image import Image
from tqdm import tqdm
from transformers import LogitsProcessorList, TemperatureLogitsWarper, TopKLogitsWarper

from mario_gpt.lm.base import BaseMarioLM
from mario_gpt.prompter import Prompter
from mario_gpt.simulator import Simulator
from mario_gpt.utils import (
    convert_level_to_png,
    load_level,
    save_level,
    trim_level,
    view_level,
)


@dataclass
class SampleOutput:
    level: Optional[List[str]]
    prompt: Optional[str] = None
    img: Optional[Image] = None
    sample_predictions_str: Optional[List[str]] = None
    sample_predictions_img: Optional[Image] = None
    level_tensor: Optional[torch.Tensor] = None
    sample_predictions_tensor: Optional[torch.Tensor] = None

    @classmethod
    def create(
        cls,
        level_tensor: torch.Tensor,
        sample_predictions_tensor: torch.Tensor,
        tokenizer,
        prompter: Optional[Prompter] = None,
    ) -> SampleOutput:
        # batch = 1
        level = None
        img = None

        try:
            level = view_level(level_tensor, tokenizer)
            img = convert_level_to_png(level)[0]
        except Exception as e:
            print(
                f"Failed to generate string or image representation for full level! Got error {e}"
            )
            level = None
            img = None
        try:
            sample_predictions_str = view_level(sample_predictions_tensor, tokenizer)
            sample_predictions_img = convert_level_to_png(sample_predictions_str)[0]
        except Exception as e:
            print(
                f"Failed to generate string or image representation for sampled predictions! Got error {e}"
            )
            sample_predictions_str = None
            sample_predictions_img = None

        prompt = None
        if prompter is not None:
            prompt = prompter(level_tensor)[0]

        return SampleOutput(
            level,
            prompt,
            img,
            sample_predictions_str,
            sample_predictions_img,
            level_tensor,
            sample_predictions_tensor,
        )

    @classmethod
    def from_level_predictions(
        cls,
        level: torch.Tensor,
        sample_predictions: torch.Tensor,
        tokenizer,
        prompter: Optional[Prompter] = None,
    ) -> Union[SampleOutput, List[SampleOutput]]:
        level_tensor = trim_level(level).squeeze().detach().cpu()
        sample_predictions_tensor = (
            trim_level(sample_predictions).squeeze().detach().cpu()
        )

        if len(level_tensor.shape) == 1:
            return SampleOutput.create(
                level_tensor, sample_predictions_tensor, tokenizer, prompter
            )

        out = []
        for _level_tensor, _sample_predictions_tensor in zip(
            level_tensor, sample_predictions_tensor
        ):
            sample_output = SampleOutput.create(
                _level_tensor, _sample_predictions_tensor, tokenizer, prompter
            )
            out.append(sample_output)
        return out

    def save(self, filename: str) -> str:
        save_level(self.level, filename)

    @classmethod
    def load(cls, filename: str) -> SampleOutput:
        level = load_level(filename)
        return SampleOutput(level=level)

    def play(self):
        simulator = Simulator(level=self.level)
        simulator.interactive()

    def run_astar(self, render=True):
        simulator = Simulator(level=self.level)
        simulator.astar(render)


class GPTSampler:
    def __init__(
        self,
        mario_lm: BaseMarioLM,
        temperature: float = 2.0,
        top_k: int = 16,
        context_len: int = 700,
        use_tqdm: bool = False,
    ):
        self.mario_lm = mario_lm
        self.temperature = temperature
        self.top_k = top_k
        self.context_len = context_len
        self.use_tqdm = use_tqdm
        self.logits_processor = LogitsProcessorList()
        self.logits_warper = LogitsProcessorList(
            [
                TopKLogitsWarper(top_k),  # number of characters
                TemperatureLogitsWarper(temperature),
            ]
        )

    @property
    def device(self) -> torch.device:
        return self.mario_lm.device

    def step(
        self,
        seed: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            attention_mask = torch.ones_like(seed).to(seed.device)
            input_ids = seed
            out = self.mario_lm.lm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                token_type_ids=None,
            )
            logits = out.logits.detach()
            if len(logits.shape) == 2:
                logits = logits.view(1, 1, -1)
            next_token_logits = logits[:, -1, :]

            next_token_scores = self.logits_processor(input_ids, next_token_logits)
            next_token_scores = self.logits_warper(input_ids, next_token_scores)
            probs = torch.nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        return next_tokens, encoder_hidden_states

    def sample(
        self,
        seed: Optional[torch.Tensor] = None,
        prompts: Optional[List[str]] = None,
        num_steps: int = 1,
        encoder_hidden_states: torch.Tensor = None,
        return_tensor: bool = False,
    ):
        context_len = self.context_len - 28
        self.mario_lm.lm.eval()
        with torch.no_grad():
            if seed is None:
                seed = (
                    self.mario_lm.tokenizer("X", return_tensors="pt")
                    .input_ids.view(1, 1)
                    .repeat(len(prompts), 1)
                )
            out_tensor = seed.to(self.device)
            if encoder_hidden_states is None:
                if prompts is not None:
                    encoder_hidden_states = torch.stack(
                        [
                            self.mario_lm.prompter.output_hidden(prompt)
                            for prompt in prompts
                        ]
                    )
                else:
                    encoder_hidden_states = torch.stack(
                        [
                            self.mario_lm.prompter(sample_prompt=True)[1]
                            for _ in range(seed.shape[0])
                        ]
                    )
            encoder_hidden_states = encoder_hidden_states.to(
                self.device
            )  # b x 1 x hidden_dim
            encoder_hidden_states = encoder_hidden_states.view(seed.shape[0], 1, -1)
            if not self.use_tqdm:
                bar = np.arange(num_steps)
            else:
                bar = tqdm(np.arange(num_steps))
            with torch.no_grad():
                for i in bar:
                    inp = out_tensor * 1
                    if len(out_tensor.shape) > 0 and out_tensor.shape[-1] > context_len:
                        diff = inp.shape[-1] % 14  # height of mario level
                        ctx = context_len + diff
                        inp = inp[:, -ctx:] * 1
                    next_tokens, encoder_hidden_states = self.step(
                        inp,
                        encoder_hidden_states=encoder_hidden_states,
                    )
                    out_tensor = torch.cat(
                        [out_tensor, next_tokens.unsqueeze(-1)], dim=-1
                    )
                    if self.use_tqdm:
                        bar.set_description(
                            f"shape: {inp.shape}, {out_tensor.shape} first: {inp[0][0]}, last: {out_tensor[0][-1]}"
                        )
            if self.use_tqdm:
                bar.close()
        sample_out = SampleOutput.from_level_predictions(
            out_tensor,
            out_tensor[:, -num_steps:],
            self.mario_lm.tokenizer,
            self.mario_lm.prompter,
        )
        self.mario_lm.lm.train()
        if return_tensor:
            return sample_out, out_tensor
        return sample_out

    def __call__(self, *args, **kwargs):
        return self.sample(*args, **kwargs)
