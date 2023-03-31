from typing import Optional, Union

from transformers import PreTrainedModel, PreTrainedTokenizer

# lm stuff
from mario_gpt.lm.base import BaseMarioLM
from mario_gpt.lm.gpt import MarioGPT
from mario_gpt.prompter import Prompter


def MarioLM(
    lm: Optional[PreTrainedModel] = None,
    tokenizer: Optional[PreTrainedTokenizer] = None,
    context_len: int = 700,
    prompter: Optional[Prompter] = None,
    lm_path: Optional[str] = None,
    tokenizer_path: Optional[str] = None,
) -> MarioGPT:
    return MarioGPT(
        lm=lm,
        tokenizer=tokenizer,
        context_len=context_len,
        prompter=prompter,
        lm_path=lm_path,
        tokenizer_path=tokenizer_path,
    )


__all__ = ["BaseMarioLM", "MarioGPT", "MarioLM"]
