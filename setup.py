from setuptools import setup, find_packages

setup(
    name="open_oasis",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A world model for generating videos using the OASIS model.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/SgtVincent/open-oasis",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "safetensors",
        "einops",
        "tqdm",
        "argparse",
        "diffusers",
        "timm",
        "av==13.1.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
