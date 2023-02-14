from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

from mario_gpt.level import FULL_LEVEL_STR_WITH_PATHS

DEFAULT_MODEL = "distilgpt2"


def split_given_size(a, size):
    return np.split(a, np.arange(size, len(a), size))


def flip_and_transpose(arr: np.array, flip_first: bool = False):
    if arr.shape[-1] > 1:
        if flip_first:
            return np.flip(arr, -1).transpose()
        return np.flip(arr.transpose(), -1)
    return arr


def join_list_of_list(str_lists):
    return ["".join(s) for s in str_lists]


def characterize(str_lists):
    return [list(s) for s in str_lists]


class MarioDataset(Dataset):
    def __init__(
        self,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        level_string: Optional[str] = None,
        context_len: int = 700,
        height: int = 14,
        remove_start_end_tokens: bool = False,
        sample_all_indices: bool = False,
    ):
        if level_string is None:
            print(
                "No level string specified, using default string FULL_LEVEL_STR_WITH_PATHS..."
            )
            level_string = FULL_LEVEL_STR_WITH_PATHS
        elif ".txt" in level_string:
            with open(level_string, "r") as file:
                level_string = file.read()

        self.character_set = set(level_string)
        if "\n" in self.character_set:
            self.character_set.remove("\n")
        self.vocab_size = len(self.character_set)
        self.sample_all_indices = sample_all_indices

        def get_training_corpus():
            yield list(level_string)

        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)

        self.tokenizer = tokenizer
        if getattr(tokenizer, "train_new_from_iterator", None) is not None:
            self.tokenizer = tokenizer.train_new_from_iterator(
                get_training_corpus(), 52000
            )
        elif getattr(tokenizer, "train_from_iterator", None) is not None:
            self.tokenizer = PreTrainedTokenizerFast(tokenizer_object=self.tokenizer)
            self.tokenizer = self.tokenizer.train_new_from_iterator(
                get_training_corpus(), self.vocab_size
            )
        self.context_len = context_len
        self.height = height

        x, self.str_arr = self.convert_level_to_tensor(level_string.split("\n"))
        self.input_ids = x["input_ids"].squeeze()
        self.attention_masks = x["attention_mask"].squeeze()
        if remove_start_end_tokens:
            self.input_ids = self.input_ids[1:-1]
            self.attention_masks = self.attention_masks[1:-1]

        self.indices = self.generate_indices()

        self.unique_tokens, self.unique_counts = self.input_ids.unique(
            return_counts=True
        )
        self.weighted_unique_counts = (
            1.0 / self.unique_counts / torch.sum(self.unique_counts)
        )

        self.token_dict = {}
        string_tokens = list(self.tokenizer.decode(self.unique_tokens))
        for int_token, string_token in zip(self.unique_tokens, string_tokens):
            self.token_dict[string_token] = int_token

    def convert_level_to_tensor(self, level: List[str]):
        str_arr = flip_and_transpose(np.array(characterize(level)))
        str_arr = "".join(join_list_of_list(str_arr))

        x = self.tokenizer(str_arr, return_tensors="pt")
        return x, str_arr

    def __len__(self):
        return self.indices.shape[0]

    def __getitem__(self, idx):
        indices = self.indices[idx]
        return self.input_ids[indices], self.attention_masks[indices]

    def generate_indices(self):
        out = []
        for idx in range(self.input_ids.shape[0] - self.context_len):
            if idx % self.height == 0 or self.sample_all_indices:
                arange = torch.arange(idx, idx + self.context_len)
                out.append(arange)
        return torch.stack(out)

    def sample_indices(self, batch_size):
        out = []
        for _ in range(batch_size):
            start_idx = np.random.randint(0, self.__len__() - self.context_len)
            indices = torch.arange(start_idx, start_idx + self.context_len)
            out.append(indices)
        return torch.stack(out)

    def __str__(self):
        str_list = characterize(self.tokenizer.batch_decode(self.x["input_ids"]))
        string = "\n".join(
            join_list_of_list(flip_and_transpose(np.array(str_list), True))
        )
        return string

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
