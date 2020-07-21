"""
Created by Sherry Chen on Jul 14, 2020
retrieve the most similar clips
"""
import os
import argparse
import pprint
import time
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
from train import load_checkpoint

num_exempler = 10
log_interval = 5
top_k = 5
split = 'val'
exempler_file = '/home/sherry/output/evaluate_exempler.txt'

def evaluate(model, test_loader, log_interval=5):
    model.eval()
    embedding = []
    vid_info = []
    with torch.no_grad():
        for batch_idx, (input, targets, info) in enumerate(test_loader):
            # if batch_idx > 1:
            #     break
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



def plot_img(cfg, data, num_exempler, row, exempler_idx, k_idx, spatial_transform=None, temporal_transform=None, output=None):
    exempler_frame = data._loading_img_path(exempler_idx, temporal_transform)
    test_frame = [data._loading_img_path(i, temporal_transform) for i in k_idx]

    exempler_title = '-'.join(exempler_frame.split('/')[-3:-2])

    print(exempler_frame)
    pprint.pprint(test_frame)
    ax = fig.add_subplot(num_exempler,len(test_frame)+1, row*(len(test_frame)+1)+1)
    image = plt.imread(exempler_frame)
    plt.imshow(image)
    ax.set_title(exempler_title, fontsize=5, pad=0.3)
    plt.axis('off')
    for i in range(len(test_frame)):
        test_title = '-'.join(test_frame[i].split('/')[-3:-2])
        ax = fig.add_subplot(num_exempler,len(test_frame)+1, row*(len(test_frame)+1)+i+2)
        image = plt.imread(test_frame[i])
        plt.imshow(image)
        ax.set_title(test_title, fontsize=5, pad=0.3)
        plt.axis('off')

    with open(os.path.join(output, 'results.txt'), 'a') as f:
        f.write('exempler_frame:\n{}\n'.format(exempler_frame))
        for frame in test_frame:
            f.write(frame)
            f.write('\n')
        f.write('\n')

    with open(os.path.join(output, 'exempler.txt'), 'a') as f:
        f.write('{}, {}'.format(exempler_idx, exempler_frame))
        f.write('\n')

def load_exempler(exempler_file):
    with open(exempler_file, 'r') as f:
        lines = f.readlines()
    exempler_idx  = []
    for line in lines:
        exempler_idx.append(int(line.split(',')[0].strip()))
    return exempler_idx

if __name__ == '__main__':
    args = parse_args()
    cfg = load_config(args)

    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    global cuda; cuda = torch.cuda.is_available()
    global device; device = torch.device("cuda" if cuda else "cpu")
    start = time.time()
    model=model_selector(cfg)
    print('=> finished generating {} backbone model...'.format(cfg.MODEL.ARCH))

    now = datetime.now()
    evaluate_output = os.path.join(args.output, 'evaluate_{}'.format(now.strftime("%d_%m_%Y_%H_%M_%S")))
    if not os.path.exists(evaluate_output):
        os.makedirs(evaluate_output)
        print('made output dir:{}'.format(evaluate_output))

    tripletnet = Tripletnet(model)
    if cuda:
        if torch.cuda.device_count() > 1:
            print("Let's use {} GPUs".format(torch.cuda.device_count()))
            tripletnet = nn.DataParallel(tripletnet)

    if args.checkpoint_path is not None:
        start_epoch, best_acc = load_checkpoint(tripletnet, args.checkpoint_path)

    model = tripletnet.module.embeddingnet

    if cuda:
        model.to(device)

    print('=> finished generating similarity network...')

    test_loader, data = data_loader.build_data_loader(split, cfg, triplets=False)
    embeddings = evaluate(model, test_loader, log_interval=log_interval)

    distance_matrix = get_distance_matrix(embeddings)

    spatial_transform = build_spatial_transformation(cfg, split)
    temporal_transform = [TemporalCenterFrame()]
    temporal_transform = TemporalCompose(temporal_transform)
    fig = plt.figure()

    if exempler_file:
        exempler_indices = load_exempler(exempler_file)
        num_exempler = len(exempler_indices)
        print('exempler_idx retrieved: {}'.format(exempler_indices))
        print('number of exemplers is: {}'.format(num_exempler))

    for i in range(num_exempler):
        if not exempler_file:
            exempler_idx = random.randint(0, distance_matrix.shape[0]-1)
        else:
            exempler_idx = exempler_indices[i]
        print('exempler video id:{}'.format(exempler_idx))
        k_idx = get_closest_data(distance_matrix, exempler_idx, top_k)
        k_nearest_data = [data[i] for i in k_idx]
        plot_img(cfg, data, num_exempler, i, exempler_idx, k_idx, spatial_transform, temporal_transform, output=evaluate_output)
    # plt.show()
    png_file = os.path.join(evaluate_output, 'plot.png')
    fig.tight_layout(pad=3.5)
    plt.savefig(png_file, dpi=300)
    print('figure saved to: {}'.format(png_file))
    print('total runtime:{}'.format(time.time()-start))
