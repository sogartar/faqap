#!/usr/bin/env python3

from subprocess import run
from os import path, mkdir
import sys
from datetime import datetime

curr_script_dir = path.abspath(path.dirname(__file__))

publish_dir = datetime.now().strftime("publish-%Y-%m-%d-%H-%M-%S.%f")
mkdir(publish_dir)

run(
    [sys.executable, path.join(curr_script_dir, "build.py")],
    check=True,
    cwd=publish_dir,
)
run(["twine", "check", "dist/*"], check=True, cwd=publish_dir)
run(["twine", "upload", "dist/*"], check=True, cwd=publish_dir)
