#!/usr/bin/env python3

from subprocess import run
from os import path

run(
    ["pip", "install", "numpy>=1.10", "scipy>=1.4", "wheel", "twine", "pre-commit"],
    check=True,
)

curr_script_dir = path.abspath(path.dirname(__file__))
run(["pre-commit", "install"], check=True, cwd=path.join(curr_script_dir, ".."))
