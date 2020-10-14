import random
from itertools import combinations

import torch
import torch.nn as nn
import torch.nn.functional as F


class OnlineTripleLoss(nn.Module):
    def __init__(self, margin, dist_metric='cosine'):
        super(OnlineTripleLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = None
        self.dist_metric = dist_metric

    # embeddings: tensor containing concatenated embeddings of anchors and positives with dim: [(batch_size * 2), dim_embedding]
    # labels: tensor containing concatenated labels of anchors and positives with dim: [(batch_size * 2)]
    def forward(self, embeddings, labels, sampling_strategy="random_negative"):

        if sampling_strategy == 'noise_contrastive':
            # Compute temperature-scaled similarity matrix 
            temperature = 0.5
            sim_matrix = 1 - pdist(embeddings, eps=0, dist_metric=self.dist_metric)
            sim_matrix.masked_fill_(torch.eye(sim_matrix.size(0), dtype=bool).cuda(), 0)
            sim_matrix = sim_matrix / temperature
            #print(sim_matrix)

            # Construct targets (i.e. list containing idx of positive for each
            # anchor)
            pos_idx_targets = torch.empty(embeddings.size(0), dtype=torch.long)
            num_anchors = embeddings.size(0)//2
            for i in range (0, embeddings.size(0)):
                pos_idx_targets[i] = (num_anchors + i) % embeddings.size(0)
            pos_idx_targets = pos_idx_targets.cuda()
            #print(pos_idx_targets)

            # Compute normalized temperature-scaled cross entropy loss (InfoNCE)
            loss = F.cross_entropy(sim_matrix, pos_idx_targets)

            return loss, 0

        else:
            # Get list of (anchor idx, postitive idx, negative idx) triplets
            self.triplet_selector = NegativeTripletSelector(self.margin, sampling_strategy, self.dist_metric)
            triplets = self.triplet_selector.get_triplets(embeddings, labels)  # list of dim: [3, batch_size]

            # Compute anchor/positive and anchor/negative distances. ap_dists and
            # an_dists are tensors with dim: [batch_size]
            if self.dist_metric == 'euclidean':
                ap_dists = F.pairwise_distance(embeddings[triplets[0], :], embeddings[triplets[1], :])
                an_dists = F.pairwise_distance(embeddings[triplets[0], :], embeddings[triplets[2], :])
            elif self.dist_metric == 'cosine':
                ap_dists = 1 - F.cosine_similarity(embeddings[triplets[0], :], embeddings[triplets[1], :], dim=1)
                an_dists = 1 - F.cosine_similarity(embeddings[triplets[0], :], embeddings[triplets[2], :], dim=1)

            # Compute margin ranking loss
            if len(triplets[0]) == 0:
                loss = torch.zeros(1, requires_grad=True)
            else:
                loss = F.relu(ap_dists - an_dists + self.margin)

            print()

            return loss.mean(), len(triplets[0])


class NegativeTripletSelector:
    def __init__(self, margin, sampling_strategy="random_negative", dist_metric='cosine'):
        super(NegativeTripletSelector, self).__init__()
        self.margin = margin
        self.sampling_strategy = sampling_strategy
        self.dist_metric = dist_metric

    # embeddings: tensor containing concatenated embeddings of anchors and positives with dim: [(batch_size * 2), dim_embedding]
    # labels: tensor containing concatenated labels of anchors and positives with dim: [(batch_size * 2)]
    def get_triplets(self, embeddings, labels):

        # Calculate distances between all embeddings to get distance_matrix
        # tensor with dim: [(batch_size * 2), (batch_size * 2)]
        distance_matrix = pdist(embeddings, eps=0, dist_metric=self.dist_metric)

        # Get tensor with unique labels (<= (batch_size * 2))
        unique_labels, counts = torch.unique(labels, return_counts=True)

        # Assert that there is no -1 (noise) label
        assert(-1 not in unique_labels)

        triplets_indices = [[] for i in range(3)]
        for i, label in enumerate(unique_labels):

            # Get embeddings indices with current label
            label_mask = labels == label
            label_indices = torch.where(label_mask)[0]
            if label_indices.shape[0] < 2:  # must have at least anchor and positive with same label
                continue

            # Get embeddings indices without current label
            negative_indices = torch.where(torch.logical_not(label_mask))[0] 
            if negative_indices.shape[0] == 0:  # must have at least one negative
                continue

            # Sample anchor/positive/negative triplet
            triplet_label_pairs = self.get_one_one_triplets(
                label_indices, negative_indices, distance_matrix,
            )
            triplets_indices[0].extend(triplet_label_pairs[0])
            triplets_indices[1].extend(triplet_label_pairs[1])
            triplets_indices[2].extend(triplet_label_pairs[2])

        return triplets_indices

    # pos_indices: tensor containing indices of embeddings with same label X
    # neg_indices: tensor containing indices of embeddings with label != X
    def get_one_one_triplets(self, pos_indices, negative_indices, dist_mat):
        triplets_indices = [[] for i in range(3)]

        # Get combinations of possible anchor/positive pairs
        # TODO: If there's > 2 pos_indices, what if 2 embeddings are from the same video?
        anchor_positives = list(combinations(pos_indices, 2))

        # For each anchor/positive pair, pick a negative and append triplet
        for i, anchor_positive in enumerate(anchor_positives):
            anchor_idx = anchor_positive[0]
            pos_idx = anchor_positive[1]

            # Compute anchor/postive dist (dim: []) and anchor/negative dists (dim: [negatives_indices.shape[0]])
            ap_dist = dist_mat[anchor_idx, pos_idx]
            an_dists = dist_mat[anchor_idx, negative_indices]

            # Sample negative index according to sampling strategy
            if self.sampling_strategy == 'random_negative':
                neg_idx = random.choice(negative_indices)
            elif self.sampling_strategy == "random_semi_hard":
                neg_list_idx = random_semi_hard_sampling(ap_dist, an_dists, self.margin)
                neg_idx = negative_indices[neg_list_idx] if neg_list_idx is not None else None
            elif self.sampling_strategy == "fixed_semi_hard":
                neg_list_idx = fixed_semi_hard_sampling(ap_dist, an_dists, self.margin)
                neg_idx = negative_indices[neg_list_idx] if neg_list_idx is not None else None
            else:
                neg_list_idx = None
                neg_idx = None

            # If failed to get semi-hard/hard negative, sample the hardest east negative instead
            if neg_idx is None: 
                neg_idx = hardest_easy_sampling(an_dists)

            triplets_indices[0].append(anchor_idx)
            triplets_indices[1].append(pos_idx)
            triplets_indices[2].append(neg_idx)
        return triplets_indices


# Return a random neg_idx giving a hard/semi-hard triplet if exists, else return None.
# Inputs: anchor/postive dist (dim: []) and anchor/negative dists (dim: [negatives_indices.shape[0]]).
# Easy triplet: d(a,p) + margin < d(a,n)  <-->  d(a,p) + margin - d(a,n) < 0
# Hard triplet: d(a,n) < d(a,p)
# Semi-hard triplet: d(a,p) < d(a,n) < d(a,p) + margin
def random_semi_hard_sampling(ap_dist, an_dists, margin):
    ap_margin_dist = ap_dist + margin
    loss = ap_margin_dist - an_dists
    possible_negs = torch.where(loss > 0)[0]

    if possible_negs.nelement() != 0:
        neg_idx = random.choice(possible_negs)
    else:
        neg_idx = None
    return neg_idx


# Return neg_idx giving the hardest hard/semi-hard triplet if exists, else return None.
# Inputs: anchor/postive dist (dim: []) and anchor/negative dists (dim: [negatives_indices.shape[0]]).
# Easy triplet: d(a,p) + margin < d(a,n)  <-->  d(a,p) + margin - d(a,n) < 0
# Hard triplet: d(a,n) < d(a,p)
# Semi-hard triplet: d(a,p) < d(a,n) < d(a,p) + margin
def fixed_semi_hard_sampling(ap_dist, an_dists, margin):
    ap_margin_dist = ap_dist + margin
    loss = ap_margin_dist - an_dists
    possible_negs = torch.where(loss > 0)[0]
    if possible_negs.nelement() != 0:
        neg_idx = torch.argmax(loss).item()
    else:
        neg_idx = None
    return neg_idx


# Return neg_idx with the smallest anchor/negative distance
def hardest_easy_sampling(an_dists):
    neg_idx = torch.argmin(an_dists).item()
    return neg_idx

# Compute distance matrix between all vectors
def pdist(vectors, eps, dist_metric):
    dist_mat = []
    for i in range(len(vectors)):
        if dist_metric=='euclidean':
            dist_mat.append(F.pairwise_distance(vectors[i], vectors, eps=eps).unsqueeze(0))
        else: #cosine
            dist_mat.append(1-F.cosine_similarity(vectors[i].unsqueeze(0), vectors, dim=1).unsqueeze(0))

    return torch.cat(dist_mat, dim=0)
