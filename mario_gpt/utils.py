import math
import os
from typing import List, Union

import numpy as np
import torch
from PIL import Image

pt = os.path.dirname(os.path.realpath(__file__))
TILE_DIR = os.path.join(pt, "data", "tiles")


def trim_level(level):
    mod = level.shape[-1] % 14
    if mod > 0:
        return level[:, :-mod]
    return level


def characterize(str_lists):
    return [list(s[::-1]) for s in str_lists]


def join_list_of_list(str_lists):
    return ["".join(s) for s in str_lists]


def view_level(level_tokens, tokenizer, flatten=False):
    if flatten:
        return tokenizer.batch_decode(level_tokens.detach().cpu().squeeze())
    str_list = tokenizer.decode(level_tokens.detach().cpu()).replace("<mask>", "Y")
    str_list = [str_list[i : i + 14] for i in range(0, len(str_list), 14)]
    for i in range(len(str_list)):
        length = len(str_list[i])
        diff = 14 - length
        if diff > 0:
            str_list[i] = str_list[i] + "Y" * diff
    return join_list_of_list(np.array(characterize(str_list)).T)


def is_flying_enemy(array, row, col):
    num_rows = array.shape[0]
    if row == num_rows - 1:
        return False
    below = array[row + 1][col]
    return below == "-"


def char_array_to_image(array, chars2pngs, target_size=None):
    """
    Convert a 16-by-16 array of integers into a PIL.Image object
    param: array: a 16-by-16 array of integers
    """
    if target_size is None:
        image = Image.new("RGB", (array.shape[1] * 16, array.shape[0] * 16))
    else:
        image = Image.new("RGB", (target_size[1] * 16, target_size[0] * 16))
    for row in range(array.shape[0]):
        for col, char in enumerate(array[row]):
            value = chars2pngs["-"]
            if char in chars2pngs:
                value = chars2pngs[char]
            else:
                print(f"REPLACING {value}", (col, row))
            image.paste(value, (col * 16, row * 16))
    return image


def convert_level_to_png(
    level: Union[str, torch.Tensor],
    tokenizer=None,
    tiles_dir: str = None,
    target_size=None,
):
    if isinstance(level, torch.Tensor):
        level = view_level(level, tokenizer)
    if tiles_dir is None:
        tiles_dir = TILE_DIR
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
        "Y": Image.fromarray(
            np.uint8(np.zeros((16, 16)))
        ),  # black square,  # self-created
        "N": Image.open(f"{tiles_dir}/N.png"),  # self-created
        "B": Image.open(f"{tiles_dir}/cannon_top.png"),
        "b": Image.open(f"{tiles_dir}/cannon_bottom.png"),
        "F": Image.open(f"{tiles_dir}/flying_koopa.png"),
    }
    levels = [list(s) for s in level]
    arr = np.array(levels)
    return char_array_to_image(arr, chars2pngs, target_size), arr, level


def generate_timelapse(level_tensor, mario_lm, interval: int = 1):
    images = []
    full_size = math.ceil(level_tensor.shape[-1] / 14)
    for i in range(1, level_tensor.shape[-1], interval):
        img = convert_level_to_png(
            level_tensor[:i], mario_lm.tokenizer, target_size=(14, full_size)
        )[0]
        images.append(img)
    return images


def save_level(level: List[str], filename: str):
    concatenated = "\n".join(level)
    with open(filename, "w") as f:
        f.write(concatenated)
    return filename


def load_level(filename: str) -> List[str]:
    with open(filename, "r") as file:
        level_string = file.read()
    lines = level_string.split("\n")
    lines = [line.strip() for line in lines]
    return lines


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
