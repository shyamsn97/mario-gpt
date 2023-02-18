import os
import subprocess
import tempfile
from typing import List, Optional

from mario_gpt.utils import load_level, save_level

pt = os.path.dirname(os.path.realpath(__file__))
JAR_PATH = os.path.join(pt, "Mario.jar")


class Simulator:
    def __init__(
        self,
        level_filename: Optional[str] = None,
        level: Optional[List[str]] = None,
        jar_path: Optional[str] = None,
    ):
        if level_filename is None and level is None:
            raise ValueError("level_filename OR level_txt must be provided!")
        elif level is None:
            level = load_level(level_filename)
        if jar_path is None:
            jar_path = JAR_PATH

        self.level_filename = level_filename
        self.level = level
        self.jar_path = jar_path

    def interactive(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=True) as t:
            save_level(self.level, t.name)

            _ = subprocess.Popen(
                ["java", "-jar", self.jar_path, t.name],
                stdout=subprocess.PIPE,
            )
