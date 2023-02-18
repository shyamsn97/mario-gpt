import os
import subprocess
import tempfile
from typing import List, Optional

from mario_gpt.utils import load_level, save_level

pt = os.path.dirname(os.path.realpath(__file__))
IMAGE_PATH = os.path.join(pt, "img/")
INTERACTIVE_JAR_PATH = os.path.join(pt, "PlayLevel.jar")
ASTAR_JAR_PATH = os.path.join(pt, "PlayAstar.jar")


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

    def astar(self, render: bool = True):
        t = tempfile.NamedTemporaryFile(suffix=".txt", delete=False)
        save_level(self.level, t.name)
        print(f"Running Astar agent on level! -- {t.name}")
        render_str = "human" if render else "norender"
        _ = subprocess.run(
            ["java", "-jar", self.astar_jar_path, t.name, render_str, IMAGE_PATH],
            stdout=subprocess.PIPE,
        )
        t.close()
        os.unlink(t.name)

    def __call__(self, simulate_mode: str = "interactive", render: bool = True):
        if simulate_mode == "interactive":
            self.interactive()
        else:
            self.astar(render)
