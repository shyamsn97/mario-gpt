import abc
from typing import Optional

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer


class BaseMarioLM(metaclass=abc.ABCMeta):
    def __init__(
        self,
        lm: Optional[PreTrainedModel] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        context_len: int = 700,
    ):
        self.context_len = context_len
        self.lm = lm

        if lm is None:
            self.lm = self.load_pretrained_lm()

        self.tokenizer = tokenizer
        if tokenizer is None:
            self.tokenizer = self.load_pretrained_tokenizer()

    @property
    def device(self):
        return self.lm.device

    def to(self, device: torch.device):
        self.lm = self.lm.to(device)
        return self

    @abc.abstractmethod
    def load_pretrained_lm(self) -> PreTrainedModel:
        """
        Model to be used in level tile prediction
        """

    @abc.abstractmethod
    def load_pretrained_tokenizer(self) -> PreTrainedTokenizer:
        """
        Tokenizer to be used to read / decode levels
        """
