from setuptools import setup, find_packages
from pv_vision import __version__

with open("long_description.md", "r") as f:
    long_description = f.read()

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setup(
    name="pv_vision",
    version=__version__,
    author="XinChen",
    author_email="chenxin0210@lbl.gov",
    description="Image analysis of defects on solar modules, including automatic detection and power loss prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hackingmaterials/pv-vision",
    packages=find_packages(),
    include_package_data=True,
    python_requires='>=3.10',
    install_requires=requirements,
    license="BSD",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
)
