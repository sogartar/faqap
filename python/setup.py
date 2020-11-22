from setuptools import setup

# read the contents of README.md
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "../README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="faqap",
    version="0.1.1",
    description="Algorithm for approximately solving quadratic assignment problems.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sogartar/faqap",
    author="sogartar",
    author_email="sogartary@yahoo.com",
    license="Unlicense",
    packages=["faqap", "faqap.test"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: The Unlicense (Unlicense)",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
    ],
    install_requires=["numpy>=1.10", "scipy>=1.4"],
    python_requires=">=3.5",
)
