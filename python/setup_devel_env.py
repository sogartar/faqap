#!/usr/bin/env python3

from subprocess import run

run(["pip", "install", "numpy>=1.10", "scipy>=1.4", "wheel", "twine"], check=True)
