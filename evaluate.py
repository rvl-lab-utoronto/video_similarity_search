"""
Created by Sherry Chen on Jul 14, 2020
retrieve the most similar clips
"""
import os
import argparse
import pprint
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics.pairwise import euclidean_distances
from datasets import data_loader
from models.triplet_net import Tripletnet
from models.model_utils import model_selector, multipathway_input
from datasets.data_loader import build_spatial_transformation
from datasets.temporal_transforms import TemporalCenterFrame
from datasets.temporal_transforms import Compose as TemporalCompose
from config.m_parser import load_config, parse_args

num_exempler = 10
log_interval = 5
top_k = 5
split = 'val'
save_dir = '/home/sherry/output/evaluate'
#
# def eval_parse_args():
#     parser = argparse.ArgumentParser(description='...')
#     parser.add_argument('--exempler', type=int, default=5)
#     parser.add_argument('--log_interval', type=int, default=5)
#     parser.add_argument('--top_k', type=int, default=5)
#     parser.add_argument('--split', type=str, default='val')
#     parser.add_argument('--output', type=str, default='/home/sherry/output/evaluate')
#
#     eargs = parser.parse_args()
#     return eargs

def evaluate(model, test_loader, log_interval=5):
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
            embedding.append(embedd.detach().cpu())
            vid_info.extend(info)
            # print('embedd size', embedd.size())
            if batch_idx % log_interval == 0:
                print('val [{}/{}]'.format(batch_idx * batch_size, len(test_loader.dataset)))

    embeddings = torch.cat(embedding, dim=0)
    return embeddings



def get_distance_matrix(embeddings):
    print('embeddings size', embeddings.size())
    embeddings = embeddings
    distance_matrix = euclidean_distances(embeddings)
    print(distance_matrix.shape)

    np.fill_diagonal(distance_matrix, float('inf'))
    return distance_matrix

def get_closest_data(distance_matrix, exempler_idx, top_k):
    test_array = distance_matrix[exempler_idx]
    idx = np.argpartition(test_array, top_k)
    top_k = idx[np.argsort(test_array[idx[:top_k]])]
    return top_k



def plot_img(cfg, data, num_exempler, row, exempler_idx, k_idx, spatial_transform=None, temporal_transform=None):
    exempler_frame = data._loading_img_path(exempler_idx, temporal_transform)
    test_frame = [data._loading_img_path(i, temporal_transform) for i in k_idx]
    print(exempler_frame)
    pprint.pprint(test_frame)
    ax = fig.add_subplot(num_exempler,len(test_frame)+1, row*(len(test_frame)+1)+1)
    image = plt.imread(exempler_frame)
    plt.imshow(image)
    plt.axis('off')
    for i in range(len(test_frame)):
        ax = fig.add_subplot(num_exempler,len(test_frame)+1, row*(len(test_frame)+1)+i+2)
        image = plt.imread(test_frame[i])
        plt.imshow(image)
        plt.axis('off')


if __name__ == '__main__':
    args = parse_args()
    cfg = load_config(args)

    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    global cuda; cuda = torch.cuda.is_available()
    global device; device = torch.device("cuda" if cuda else "cpu")

    model=model_selector(cfg)
    print('=> finished generating {} backbone model...'.format(cfg.MODEL.ARCH))

    # Load pretrained backbone if path exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if args.pretrain_path is not None:
        model = load_pretrained_model(model, args.pretrain_path)

    if cuda:
        if torch.cuda.device_count() > 1:
            print("Let's use {} GPUs".format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)
    print('=> finished generating similarity network...')

    test_loader, data = data_loader.build_data_loader(split, cfg, triplets=False)
    embeddings = evaluate(model, test_loader, log_interval=log_interval)

    distance_matrix = get_distance_matrix(embeddings)

    spatial_transform = build_spatial_transformation(cfg, split)
    temporal_transform = [TemporalCenterFrame()]
    temporal_transform = TemporalCompose(temporal_transform)
    fig = plt.figure()
    for i in range(num_exempler):
        exempler_idx = random.randint(0, distance_matrix.shape[0]-1)
        print('exempler video id:{}'.format(exempler_idx))
        k_idx = get_closest_data(distance_matrix, exempler_idx, top_k)
        k_nearest_data = [data[i] for i in k_idx]
        plot_img(cfg, data, num_exempler, i, exempler_idx, k_idx, spatial_transform, temporal_transform)
    # plt.show()
    now = datetime.now()
    png_file = os.path.join(save_dir, 'evaluate_{}.png'.format(now.strftime("%d_%m_%Y_%H_%M_%S")))
    plt.savefig(png_file)
    print('figure saved to: {}'.format(png_file))
