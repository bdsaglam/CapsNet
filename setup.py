from setuptools import setup, find_packages

setup(
    name='capsnet',
    version=0.1,
    packages=find_packages(),
    install_requires=[
        'numpy',
        'torch',
        'torchvision',
        'tqdm'
    ],
    description="",
    author="BDS",
)
