"""
Created by Sherry Chen on Jul 14, 2020
retrieve the most similar clips
"""
import os
import pprint
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics.pairwise import euclidean_distances
from datasets import data_loader
from models.triplet_net import Tripletnet
from models.model_utils import model_selector, multipathway_input
from config.m_parser import load_config, parse_args

log_interval = 5
top_k = 5

def evaluate(model, test_loader):
    model.eval()
    embedding = []
    vid_info = []
    with torch.no_grad():
        for batch_idx, (input, targets, info) in enumerate(test_loader):
            if batch_idx > 1:
                break

            batch_size = input.size(0)

            if cfg.MODEL.ARCH == 'slowfast':
                input = multipathway_input(input, cfg)
                if cuda:
                    for i in range(len(input)):
                        input= input[i].to(device)
            else:
                if cuda:
                    input= input.to(device)

            embedd = model(input)
            embedding.append(embedd)
            vid_info.extend(info)
            print('embedd size', embedd.size())
            if batch_idx % log_interval == 0:
                print('val [{}/{}]'.format(batch_idx * batch_size, len(test_loader.dataset)))

    embeddings = torch.cat(embedding, dim=0)
    print('embeddings size', embeddings.size())
    embeddings = embeddings.cpu()
    distance_matrix = euclidean_distances(embeddings)
    print(distance_matrix.shape)

    np.fill_diagonal(distance_matrix, float('inf'))
    exempler_idx = random.randint(0, distance_matrix.shape[0])
    print('exempler video id:{}'.format(exempler_idx))
    test_array = distance_matrix[exempler_idx]

    idx = np.argpartition(test_array, top_k)
    print('top k indices',idx[:top_k])
    print('print nearest k distances', test_array[idx[:top_k]])
    print('exempler vid:')
    print(vid_info[exempler_idx])
    k_nearest = test_array[idx[:top_k]]
    k_nearest_path = [vid_info[id] for id in idx[:top_k]]
    print('closes_vid:')
    pprint.pprint(k_nearest_path)

if __name__ == '__main__':
    args = parse_args()
    cfg = load_config(args)

    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    global cuda; cuda = torch.cuda.is_available()
    global device; device = torch.device("cuda" if cuda else "cpu")

    model=model_selector(cfg)
    print('=> finished generating {} backbone model...'.format(cfg.MODEL.ARCH))

    # Load pretrained backbone if path exists
    if args.pretrain_path is not None:
        model = load_pretrained_model(model, args.pretrain_path)

    if cuda:
        if torch.cuda.device_count() > 1:
            print("Let's use {} GPUs".format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)
    print('=> finished generating similarity network...')

    test_loader = data_loader.build_data_loader('val', cfg, triplets=False)
    evaluate(model, test_loader)
