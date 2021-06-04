
import sys,os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
import torch
import torchvision.models as models
import cv2
import numpy as np
from torchvision import transforms
from sklearn.cluster import AgglomerativeClustering, DBSCAN, OPTICS, KMeans
from sklearn.metrics import normalized_mutual_info_score
import time
import pickle as pkl

from config.m_parser import load_config, arg_parser
from datasets import data_loader
from datasets.spatial_transforms import (Compose, Resize, CenterCrop, ToTensor)

# from https://github.com/jasonlaska/spherecluster, install with:
# `python setup.py install`
# requires scikit-learn==0.22.0
from spherecluster import SphericalKMeans  # commented out to remove warnings

from finch import FINCH

np.random.seed(1)

# Preprocessing for kmeans - l2normalize
def preprocess_features_kmeans(data):
    l2norms = torch.norm(data, dim=1, keepdim=True)
    data = data / l2norms
    print('l2-normalized data')
    return data


# Perform clustering 
def fit_cluster(embeddings, method='Agglomerative', k=1000, l2normalize=True,
        finch_partition=0):


    assert(method in ['DBSCAN', 'Agglomerative', 'OPTICS', 'kmeans',
        'spherical_kmeans', 'finch'])

    print("Clustering with {}...".format(method))
    if method == 'kmeans':
        print("k:", k)

    if method == 'Agglomerative':
        distance_threshold = 0.24 #0.24 for ucf train
        trained_cluster_obj = AgglomerativeClustering(n_clusters=None,
                                                      linkage='average',
                                                      distance_threshold=distance_threshold,
                                                      affinity='cosine').fit(embeddings)

    elif method == 'DBSCAN':
        # If small clusters have too many incorrect increase min_samples, if there are some very large clusters with
        # too many incorrect decrease eps, if too few / little cluster representation decrease min_samples, if too many -1 increase eps
        trained_cluster_obj = DBSCAN(eps=0.14, # 0.18 for ucf val, #0.14 for ucf train, #0.12 for kin train
                                     min_samples=2, #3 for ucf val, #2 for ucf train, 3 for kin train
                                     metric='cosine',
                                     n_jobs=-1).fit(embeddings)

    elif method == 'kmeans':
        #pre-process - l2 normalize embeddings
        if l2normalize:
            embeddings = preprocess_features_kmeans(embeddings)

        n_clusters = k #2000 for ucf train
        trained_cluster_obj = KMeans(n_clusters=n_clusters,
                                     n_init=10).fit(embeddings)

    elif method == 'spherical_kmeans':
        n_clusters = k
        print('clustering with spherical kmeans with k={}'.format(n_clusters))
        print(embeddings.shape)
        trained_cluster_obj = SphericalKMeans(n_clusters=n_clusters).fit(embeddings)

    elif method == 'finch':
        embeddings = embeddings.detach().cpu().numpy()
        c, num_clust, req_c = FINCH(embeddings, distance='cosine')

        PARTITION = finch_partition
        labels = c[:,PARTITION]
        n_clusters = num_clust[PARTITION]
        print('Taking partition {} from finch'.format(PARTITION))

    elif method == 'OPTICS':
        trained_cluster_obj = OPTICS(min_samples=3, max_eps=0.20, cluster_method='dbscan', metric='cosine', n_jobs=-1).fit(embeddings)


    if method != 'finch':
        labels = trained_cluster_obj.labels_
        print(labels.shape)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    print("Fitted " + str(n_clusters) + " clusters with " + str(method))
    return labels




