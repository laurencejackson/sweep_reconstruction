from setuptools import setup

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
    name="sweep_reconstruction",
    version='1.0',
    description='Python code for reconstructing 2D dynamic linearly order MRI data with respiration motion correction',
    author='Laurence Jackson',
    author_email='Laurence.Jackson@kcl.ac.uk',
    url='https://github.com/laurencejackson/sweep_reconstruction',
    packages=['sweeprecon'],
    long_description=long_description
)