"""
Created by Sherry Chen on Jul 3, 2020
"""

import torch
import torch.nn as nn


class Tripletnet(nn.Module):
    def __init__(self, embeddingnet, dist_metric):
        super(Tripletnet, self).__init__()
        self.embeddingnet = embeddingnet

        assert dist_metric in ['cosine', 'euclidean']
        if dist_metric == 'euclidean':
            self.dist_fcn = nn.PairwiseDistance(p=2)
        elif dist_metric == 'cosine':
            self.dist_fcn = nn.CosineSimilarity(dim=1, eps=1e-8)


    def forward(self, x, y, z):
        # print('forwarding...')
        embedded_x = self.embeddingnet(x)
        embedded_y = self.embeddingnet(y)
        embedded_z = self.embeddingnet(z)
        if dist_metric == 'euclidean':
            dist_a = self.dist_fcn(embedded_x, embedded_y)
            dist_b = self.dist_fcn(embedded_x, embedded_z)
        elif dist_metric == 'cosine':
            dist_a = 1 - self.dist_fcn(embedded_x, embedded_y)
            dist_b = 1 - self.dist_fcn(embedded_x, embedded_z)
        return dist_a, dist_b, embedded_x, embedded_y, embedded_z
