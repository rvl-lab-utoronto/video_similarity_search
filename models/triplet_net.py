"""
Created by Sherry Chen on Jul 3, 2020
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Tripletnet(nn.Module):
    def __init__(self, embeddingnet, dist_metric='cosine'):
        super(Tripletnet, self).__init__()
        self.embeddingnet = embeddingnet

        assert dist_metric in ['cosine', 'euclidean']
        self.dist_metric = dist_metric


    def forward(self, x, y, z):
        # print('forwarding...')
        embedded_x = self.embeddingnet(x)
        embedded_y = self.embeddingnet(y)
        embedded_z = self.embeddingnet(z)
        if isinstance(embedded_x,tuple):
            embedded_x = embedded_x[0]
        if isinstance(embedded_y, tuple):
            embedded_y = embedded_y[0]
        if isinstance(embedded_z, tuple):
            embedded_z = embedded_z[0]

        if self.dist_metric == 'euclidean':
            dist_a = F.pairwise_distance(embedded_x, embedded_y, 2)
            dist_b = F.pairwise_distance(embedded_x, embedded_z, 2)
        elif self.dist_metric == 'cosine':
            dist_a = 1 - F.cosine_similarity(embedded_x, embedded_y, dim=1)
            dist_b = 1 - F.cosine_similarity(embedded_x, embedded_z, dim=1)
        return dist_a, dist_b, embedded_x, embedded_y, embedded_z
