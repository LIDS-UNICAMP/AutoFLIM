from setuptools import setup, find_packages
import subprocess

install_deps = [
        "random2",
        "pillow>=10.2.0",
        "numpy>=1.26.4",
        "torch>=2.5.1",
        "torchvision>=0.20.1",
        "scikit-image>=0.25.2",
        "scikit-learn>=1.6.1",
        "faiss-cpu>=1.10.0",
        "matplotlib",
        "monai>=1.4.0",
    ]

with open("README.md", "r") as fh:
    long_description = fh.read()

__version__ = None
exec(open("pyflim/_version.py").read())

setup(
    name='pyflim',
    version='0.1',
    packages=find_packages(),
    install_requires=install_deps,
    author='Leonardo de Melo João',
    author_email='leomelo168@gmail.com',
    maintainer='Gilson Júnior Soares',
    maintainer_email='your@email',
    description='This is a python implementation of the Feature Learning from Image Markers.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/LIDS-UNICAMP/flim-python',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 2 - Pre-Alpha",
        "Environment :: GPU :: NVIDIA CUDA",
        "Intended Audience :: Education :: Developers",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering :: \
            Artificial Intelligence :: Image Recognition",
        "Topic :: Software Development :: Libraries",
    ],
    python_requires='>=3.10',
)
