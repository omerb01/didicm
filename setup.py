"""
Setup script for didicm package.
"""

from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Install dependencies from requirements.txt
requirements = []
requirements_file = this_directory / "requirements.txt"
if requirements_file.exists():
    requirements = [
        req.strip()
        for req in requirements_file.read_text(encoding="utf-8").strip().split("\n")
        if req.strip() and not req.startswith("#")
    ]

setup(
    name="didicm",
    version="0.1.0",
    author="Omer Belhasin",
    author_email="omerb01@gmail.com",
    description="Discrete Diffusion Classification Modeling for Image Classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/omerb01/didicm",
    packages=find_packages(),
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    include_package_data=True,
    zip_safe=False,
    keywords="diffusion, classification, image-classification, deep-learning, machine-learning",
)

