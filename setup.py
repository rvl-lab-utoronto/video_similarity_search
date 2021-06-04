import os
from setuptools import setup, find_packages

setup(
    name = "video_similarity_search",
    version = "1.0",
    packages=find_packages(),
    install_requires=['numpy',
                    'scipy',
                    'pandas',
                    'matplotlib',
                    'scikit-learn',
                    'opencv-python',
                    'torch',
                    'torchvision',
                    'tensorboard',
                    'joblib',
                    'tqdm',
                    'ipdb',
                    'h5py',
                    'simplejson',
                    'fvcore',
                    'gspread',
                    'oauth2client',
                    'apiclient',
                    'google-api-python-client',
                    'torchviz'
                    ], #external packages as dependencies
    scripts=[
            'train.py',
            'evaluate.py',
            'visualize.py'
           ]
)
