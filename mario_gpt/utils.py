from typing import Union

import numpy as np
import torch
from PIL import Image


def characterize(str_lists):
    return [list(s[::-1]) for s in str_lists]


def join_list_of_list(str_lists):
    return ["".join(s) for s in str_lists]


def view_level(level_tokens, tokenizer):
    str_list = [
        s.replace("<mask>", "Y")
        for s in tokenizer.batch_decode(level_tokens.detach().cpu().view(-1, 14))
    ]
    return join_list_of_list(np.array(characterize(str_list)).T)


def is_flying_enemy(array, row, col):
    num_rows = array.shape[0]
    if row == num_rows - 1:
        return False
    below = array[row + 1][col]
    return below == "-"


def char_array_to_image(array, chars2pngs):
    """
    Convert a 16-by-16 array of integers into a PIL.Image object
    param: array: a 16-by-16 array of integers
    """
    image = Image.new("RGB", (array.shape[1] * 16, array.shape[0] * 16))
    for row in range(array.shape[0]):
        for col, char in enumerate(array[row]):
            value = chars2pngs["-"]
            # if char == "E":
            #     if is_flying_enemy(array, row, col):
            #         char = "F"
            if char in chars2pngs:
                value = chars2pngs[char]
            else:
                print(f"REPLACING {value}", (col, row))

            image.paste(value, (col * 16, row * 16))
    return image


def convert_level_to_png(
    level: Union[str, torch.Tensor], tiles_dir: str, tokenizer=None
):
    if isinstance(level, torch.Tensor):
        level = view_level(level, tokenizer)
    chars2pngs = {
        "-": Image.open(f"{tiles_dir}/smb-background.png"),
        "X": Image.open(f"{tiles_dir}/smb-unpassable.png"),
        "S": Image.open(f"{tiles_dir}/smb-breakable.png"),
        "?": Image.open(f"{tiles_dir}/smb-question.png"),
        "Q": Image.open(f"{tiles_dir}/smb-question.png"),
        "o": Image.open(f"{tiles_dir}/smb-coin.png"),
        "E": Image.open(f"{tiles_dir}/smb-enemy.png"),
        "<": Image.open(f"{tiles_dir}/smb-tube-top-left.png"),
        ">": Image.open(f"{tiles_dir}/smb-tube-top-right.png"),
        "[": Image.open(f"{tiles_dir}/smb-tube-lower-left.png"),
        "]": Image.open(f"{tiles_dir}/smb-tube-lower-right.png"),
        "x": Image.open(f"{tiles_dir}/smb-path.png"),  # self-created
        "Y": Image.open(f"{tiles_dir}/Y.png"),  # self-created
        "N": Image.open(f"{tiles_dir}/N.png"),  # self-created
        "B": Image.open(f"{tiles_dir}/cannon_top.png"),
        "b": Image.open(f"{tiles_dir}/cannon_bottom.png"),
        "F": Image.open(f"{tiles_dir}/flying_koopa.png"),
    }
    levels = [list(s) for s in level]
    arr = np.array(levels)
    return char_array_to_image(arr, chars2pngs), arr, level


TOKENS = [
    "-",
    "X",
    "S",
    "?",
    "Q",
    "o",
    "E",
    "<",
    ">",
    "[",
    "]",
    "x",
    "Y",
    "N",
    "B",
    "b",
]
