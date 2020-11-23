#!/usr/bin/env python3

from subprocess import run
from os import path
import sys

curr_script_dir = path.abspath(path.dirname(__file__))

run(
    [
        sys.executable,
        path.join(curr_script_dir, "setup.py"),
        "bdist_wheel",
        "--universal",
    ],
    check=True,
)
run(
    [sys.executable, path.join(curr_script_dir, "setup.py"), "clean", "--all"],
    check=True,
)
