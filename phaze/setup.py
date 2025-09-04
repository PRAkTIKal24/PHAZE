#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='phaze',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'torch',
        'numpy',
        'scipy',
        'uv',
        'onnx',
        'onnxruntime',
        'ezkl',
    ],
    extras_require={
        'dev': [
            'pytest',
            'black',
            'isort',
            'flake8',
            'sphinx',
            'sphinx_rtd_theme',
            'myst-parser',
        ],
    },
    author='Manus AI',
    description='A Python package for PHAZE: cryptographic and ZKML-based low latency inference at LHC',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/your_username/phaze', # Placeholder, update with actual repo URL
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
)


