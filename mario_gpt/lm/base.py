import abc
import os
from typing import Any, Dict, Optional

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer


class BaseMarioLM(metaclass=abc.ABCMeta):

    PRETRAINED_LM_PATH = ""
    PRETRAINED_TOKENIZER_PATH = ""

    BASE_LM_PATH = ""
    BASE_TOKENIZER_PATH = ""

    def __init__(
        self,
        lm: Optional[PreTrainedModel] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        context_len: int = 700,
        lm_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        lm_kwargs: Dict[str, Any] = {},
        tokenizer_kwargs: Dict[str, Any] = {},
    ):
        self.load_pretrained(
            lm_path, tokenizer_path, lm, tokenizer, lm_kwargs, tokenizer_kwargs
        )
        self.context_len = context_len

    def train(self):
        self.lm.train()

    def eval(self):
        self.lm.eval()

    @property
    def device(self):
        return self.lm.device

    def to(self, device: torch.device):
        self.lm = self.lm.to(device)
        return self

    def save_model(self, checkpoint_path: str, it: int):
        self.lm.save_pretrained(os.path.join(checkpoint_path, f"iteration_{it}"))

    @abc.abstractmethod
    def load_pretrained_lm(
        self, path: str, lm_kwargs: Dict[str, Any]
    ) -> PreTrainedModel:
        """
        Model to be used in level tile prediction
        """

    @abc.abstractmethod
    def load_pretrained_tokenizer(
        self, path: str, tokenizer_kwargs: Dict[str, Any]
    ) -> PreTrainedTokenizer:
        """
        Tokenizer to be used to read / decode levels
        """

    def load_pretrained(
        self,
        lm_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        lm: Optional[PreTrainedModel] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        lm_kwargs: Dict[str, Any] = {},
        tokenizer_kwargs: Dict[str, Any] = {},
    ):
        self.lm = lm
        self.tokenizer = tokenizer

        if lm is None:
            if lm_path is None:
                lm_path = self.PRETRAINED_LM_PATH

            print(f"Using {lm_path} lm")
            self.lm = self.load_pretrained_lm(lm_path, lm_kwargs)

        if tokenizer is None:
            if tokenizer_path is None:
                tokenizer_path = self.PRETRAINED_LM_PATH

            print(f"Using {tokenizer_path} tokenizer")
            self.tokenizer = self.load_pretrained_tokenizer(
                tokenizer_path, tokenizer_kwargs
            )
