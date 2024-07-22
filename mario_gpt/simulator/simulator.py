import os
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from mario_gpt.utils import load_level, save_level

pt = os.path.dirname(os.path.realpath(__file__))
IMAGE_PATH = os.path.join(pt, "img/")
INTERACTIVE_JAR_PATH = os.path.join(pt, "PlayLevel.jar")
ASTAR_JAR_PATH = os.path.join(pt, "PlayAstar.jar")


def load_images(directory):
    images = []
    names = os.listdir(directory)
    for i in range(len(names)):
        p = os.path.abspath(os.path.join(directory, f"img{i}.png"))
        im = Image.open(p)
        images.append(im.copy())
        im.close()
    return images


def read_observations(filename: str):
    levels = []
    with open(filename) as f:
        for line in f:
            levels.append(np.transpose(eval(line.replace("\n", ""))).astype(str))
    return levels


def read_actions(filename: str):
    levels = []
    with open(filename) as f:
        for line in f:
            levels.append(line.replace("\n", ""))
    return levels


tile_conversion = {
    "0": " ",
    "1": "E",
    "17": "X",
    "22": "S",
    "24": "?",
    "19": "B",
    "20": "B",
    "30": "X",
    "31": "o",
    "34": "P",
    "35": "P",
    "36": "P",
    "37": "P",
    "59": "x",
    "77": "M",
    "55": "G",
    "56": "G",
}


def get_asciis(levels):
    import copy

    np_levels = []
    out = []
    for idx, l in enumerate(levels):
        lev = copy.deepcopy(l)
        ascii_level = []
        for i in range(lev.shape[0]):
            for j in range(lev.shape[1]):
                if lev[i][j] not in tile_conversion:
                    print(idx, "NOT IN", tile_conversion, "IDX", idx)
                lev[i][j] = tile_conversion.get(lev[i][j])
            ascii_level.append(" ".join(lev[i]))
        np_levels.append(lev)
        out.append("\n".join(ascii_level))
    return out, np_levels


def text_to_image(
    ascii_art, font_size=10, text_color=(255, 255, 255), background_color=(0, 0, 0)
):
    # Split the ASCII art into lines
    lines = ascii_art.split("\n")

    # Determine image dimensions
    width = max(len(line) for line in lines)
    height = len(lines)

    # Create a blank PIL image with transparent background
    image = Image.new("RGBA", (width * font_size, height * font_size), background_color)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()  # You can change the font if needed

    # Draw the ASCII characters onto the image
    y = 0
    for line in lines:
        x = 0
        for char in line:
            if char != " ":  # Only draw if the character is not a space
                draw.text((x, y), char, font=font, fill=text_color)
            x += font_size / 2
        y += font_size

    return image


def make_video(imgs, output_path="output.mp4"):
    import imageio
    import numpy as np

    # List of PIL images
    # Output file path
    # Create video writer object
    writer = imageio.get_writer(output_path, fps=30)  # Adjust the fps as needed

    # Iterate over images and add them to the video
    for image in imgs:
        # Convert PIL image to numpy array
        image_np = np.array(image)

        # Add image to video
        writer.append_data(image_np)

    # Close the writer
    writer.close()

    print("Video created successfully!")


def concatenate_images_horizontally(image1, image2):
    # Calculate the dimensions of the new image
    width = image1.width + image2.width
    height = max(image1.height, image2.height)

    # Create a blank image with the calculated dimensions
    concatenated_image = Image.new("RGB", (width, height))

    # Paste the first image at the leftmost position
    concatenated_image.paste(image1, (int(image2.width / 2), 10))

    # Paste the second image to the right of the first image
    concatenated_image.paste(image2, (image1.width, 0))

    return concatenated_image


@dataclass
class SimulatorOutput:
    images: List[Any]
    observations: Any
    one_hot_obs: Any
    np_obs: Any
    actions: Any
    game_status: bool
    console_out: str

    def make_timelapse(self, filename="output.mp4"):
        images = self.images
        observations = self.observations
        out = []
        for i, o in zip(images, observations):
            out.append(
                concatenate_images_horizontally(text_to_image(o, font_size=16), i)
            )
        make_video(out, filename)


class Simulator:
    def __init__(
        self,
        level_filename: Optional[str] = None,
        level: Optional[List[str]] = None,
        interactive_jar_path: Optional[str] = None,
        astar_jar_path: Optional[str] = None,
    ):
        if level_filename is None and level is None:
            raise ValueError("level_filename OR level_txt must be provided!")
        elif level is None:
            level = load_level(level_filename)
        if interactive_jar_path is None:
            interactive_jar_path = INTERACTIVE_JAR_PATH
        if astar_jar_path is None:
            astar_jar_path = ASTAR_JAR_PATH

        self.level_filename = level_filename
        self.level = level
        self.interactive_jar_path = interactive_jar_path
        self.astar_jar_path = astar_jar_path

    def interactive(self):
        t = tempfile.NamedTemporaryFile(suffix=".txt", delete=False)
        save_level(self.level, t.name)
        print(f"Playing level interactively -- {t.name}!")
        _ = subprocess.run(
            ["java", "-jar", self.interactive_jar_path, t.name, IMAGE_PATH],
            stdout=subprocess.PIPE,
        )
        t.close()
        os.unlink(t.name)

    def astar(
        self,
        render: bool = True,
        image_path: Optional[str] = None,
    ):
        if image_path is None:
            image_path = IMAGE_PATH
        with tempfile.TemporaryDirectory() as tmpdirname:
            output_image_path = os.path.join(tmpdirname, "images")
            os.makedirs(output_image_path)

            observations_path = os.path.join(tmpdirname, "observations.txt")
            actions_path = os.path.join(tmpdirname, "actions.txt")

            level_path = os.path.join(tmpdirname, "level.txt")
            save_level(self.level, level_path)
            print(f"Running Astar agent on level! -- {level_path}")
            render_str = "human" if render else "norender"
            subprocess_output = subprocess.run(
                [
                    "java",
                    "-jar",
                    self.astar_jar_path,
                    level_path,
                    render_str,
                    image_path,
                    tmpdirname,
                ],
                stdout=subprocess.PIPE,
            )
            console_out = subprocess_output.stdout.splitlines()
            images = load_images(output_image_path)
            one_hot_obs = read_observations(observations_path)
            observations, np_obs = get_asciis(one_hot_obs)
            actions = read_actions(actions_path)
            console_out = [c.decode("utf-8") for c in console_out]
            game_status = "WIN" in "\n".join(console_out).split("Game Status")[-1]
            return SimulatorOutput(
                images=images,
                observations=observations,
                actions=actions,
                np_obs=np_obs,
                one_hot_obs=one_hot_obs,
                game_status=game_status,
                console_out=console_out,
            )

    def __call__(self, simulate_mode: str = "interactive", render: bool = True):
        if simulate_mode == "interactive":
            self.interactive()
        else:
            self.astar(render)
