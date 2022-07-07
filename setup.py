from setuptools import setup, find_packages
from pv_vision import __version__

with open("long_description.md", "r") as f:
    long_description = f.read()

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
    python_requires='>=3.8',
    install_requires=[
        "matplotlib>=3.4.2",
        "opencv_python>=4.4.0.44",
        "seaborn>=0.11.2",
        "torchvision>=0.7.0",
        "pandas>=1.3.2",
        "imutils>=0.5.3",
        "scipy>=1.6.0",
        "numpy>=1.19.2",
        "tqdm>=4.56.0",
        "torch>=1.6.0",
        "Pillow>=9.0.1",
        "scikit_learn>=0.24.2",
    ],
    license="BSD",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
)
