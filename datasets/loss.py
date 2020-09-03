import random
from itertools import combinations

import torch
import torch.nn as nn
import torch.nn.functional as F


class OnlineTripleLoss(nn.Module):
    def __init__(self, margin):
        super(OnlineTripleLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = None #NegativeTripletSelector(margin, sampling_strategy)

    def forward(self, embeddings, labels, sampling_strategy="random_negative"):
        self.triplet_selector = NegativeTripletSelector(self.margin, sampling_strategy)
        triplets = self.triplet_selector.get_triplets(embeddings, labels)

        ap_dists = F.pairwise_distance(
            embeddings[triplets[0], :], embeddings[triplets[1], :]
        )
        an_dists = F.pairwise_distance(
            embeddings[triplets[0], :], embeddings[triplets[2], :]
        )
        loss = F.relu(ap_dists - an_dists + self.margin)
        return loss.mean(), len(triplets[0])


class NegativeTripletSelector:
    def __init__(self, margin, sampling_strategy="random_negative"):
        super(NegativeTripletSelector, self).__init__()
        self.margin = margin
        self.sampling_strategy = sampling_strategy

    def get_triplets(self, embeddings, labels):
        distance_matrix = pdist(embeddings, eps=0)
        unique_labels, counts = torch.unique(labels, return_counts=True)
        print('labels', labels)
        print('unique_labels', unique_labels, counts)

        triplets_indices = [[] for i in range(3)]
        for i, label in enumerate(unique_labels):
            label_mask = labels == label
            label_indices = torch.where(label_mask)[0]
            if label_indices.shape[0] < 2:
                continue
            negative_indices = torch.where(torch.logical_not(label_mask))[0]
            triplet_label_pairs = self.get_one_one_triplets(
                label_indices, negative_indices, distance_matrix,
            )

            triplets_indices[0].extend(triplet_label_pairs[0])
            triplets_indices[1].extend(triplet_label_pairs[1])
            triplets_indices[2].extend(triplet_label_pairs[2])
            # print('triplets', triplet_label_pairs)
        return triplets_indices

    def get_one_one_triplets(self, pos_labels, negative_indices, dist_mat):
        # print('pos_labels', pos_labels)
        anchor_positives = list(combinations(pos_labels, 2))
        # print('anchor', anchor_positives)
        triplets_indices = [[] for i in range(3)]
        for i, anchor_positive in enumerate(anchor_positives):
            anchor_idx = anchor_positive[0]
            pos_idx = anchor_positive[1]
            ap_dist = dist_mat[anchor_idx, pos_idx]
            an_dists = dist_mat[anchor_idx, negative_indices]

            if self.sampling_strategy == 'random_negative':
                print('random_negative')
                print('negative_indices', negative_indices)
                neg_idx = random.choice(negative_indices)

            elif self.sampling_strategy == "random_semi_hard":
                neg_list_idx = random_semi_hard_sampling(ap_dist, an_dists, self.margin)
                neg_idx = negative_indices[neg_list_idx]

            elif self.sampling_strategy == "fixed_semi_hard":
                neg_list_idx = fixed_semi_hard_sampling(ap_dist, an_dists, self.margin)
                neg_idx = negative_indices[neg_list_idx]
            else:
                neg_list_idx = None
                neg_idx = None

            if neg_idx is None:
                neg_idx = random.choice(negative_indices)

            print('neg idx', neg_idx)
            triplets_indices[0].append(anchor_idx)
            triplets_indices[1].append(pos_idx)
            triplets_indices[2].append(neg_idx)
            print('triplets', triplets_indices)
        return triplets_indices


def random_semi_hard_sampling(ap_dist, an_dists, margin):
    ap_margin_dist = ap_dist + margin
    loss = ap_margin_dist - an_dists
    possible_negs = torch.where(loss > 0)[0]
    # print('negs',possible_negs)

    if possible_negs.nelement() != 0:
        neg_idx = random.choice(possible_negs)
    else:
        neg_idx = None
    return neg_idx


def fixed_semi_hard_sampling(ap_dist, an_dists, margin):
    ap_margin_dist = ap_dist + margin
    loss = ap_margin_dist - an_dists
    possible_negs = torch.where(loss > 0)[0]
    if possible_negs.nelement() != 0:
        neg_idx = torch.argmax(loss).item()
    else:
        neg_idx = None
    # neg_idx = torch.argmin(an_dists).item()
    return neg_idx


def pdist(vectors, eps):
    dist_mat = []
    for i in range(len(vectors)):
        dist_mat.append(
            F.pairwise_distance(vectors[i], vectors, eps=eps).unsqueeze(0)
        )
    return torch.cat(dist_mat, dim=0)
