import os
from setuptools import setup, find_packages

setup(
    name = "video_similarity_search",
    version = "1.0",
    author = "Sherry Chen, Salar Hosseini",
    author_email = "yuxuansherry.chen@mail.utoronto.ca, salar.hosseinikhorasgani@mail.utoronto.ca",
    description = ("one-shot exemplar-based visual search for video events"),
    url = "https://github.com/rvl-lab-utoronto/video_similarity_search.git",
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
