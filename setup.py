#!/usr/bin/env python

try:
    from setuptools import setup
except ImportError:
    try:
        from ez_setup import use_setuptools
        use_setuptools()
        from setuptools import setup
    except Exception as e:
        print("Forget setuptools, trying distutils...")
        from distutils.core import setup


description = ("Tools used for computational neuroscience research "
               + "at the University of Waterloo's "
               + "Computational Neuroscience Research Group.")
setup(
    name="mytools",
    version="0.0.1",
    author="Eric Crawford",
    author_email="e2crawfo@uwaterloo.ca",
    packages=['mytools'],
    scripts=[],
    url="https://github.com/e2crawfo/mytools",
    description=description,
    requires=[
        "numpy (>=1.5.0)",
    ],
)
