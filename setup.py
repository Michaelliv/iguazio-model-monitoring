from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

setup(
    name='modelmonitoring',  # Required
    version='0.0.1',  # Required
    packages=find_packages(),  # Required
    python_requires='>=3.6, <4',
    # install_requires=[],  # Optional
)