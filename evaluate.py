"""
Created by Sherry Chen on Jul 14, 2020
retrieve the most similar clips
"""
import os
import argparse
import pprint
import time
import numpy as np
import torch
import cv2
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
from datasets import data_loader
from models.triplet_net import Tripletnet
from models.model_utils import model_selector, multipathway_input
from datasets.data_loader import build_spatial_transformation
from datasets.temporal_transforms import TemporalCenterFrame, TemporalSpecificCrop
from datasets.temporal_transforms import Compose as TemporalCompose
from config.m_parser import load_config, arg_parser
from train import load_checkpoint
from misc.upload_gdrive import GoogleDriveUploader

# num_exemplar = 10
log_interval = 10
top_k = 5
split = 'val'
exemplar_file = None
#exemplar_file = '/home/sherry/output/u_exemplar.txt'
# np.random.seed(1)


# Argument parser
def m_arg_parser(parser):
    parser.add_argument(
        '--root_dir',
        type=str,
        default='.'
    )
    parser.add_argument(
        '--name',
        type=str,
        default=None,
        help='Please specify the name (e.g. ResNet18_K, SlowFast_U): '
    )
    parser.add_argument(
        '--num_exemplar',
        type=int,
        default=None,
        help='Please specify number of exemplar videos: '
    )
    parser.add_argument(
        '--heatmap',
        action='store_true',
        help='Run temporal heatmap visualization'
    )
    parser.add_argument(
        "--ex_idx",
        default=None,
        type=int,
        help='Exemplar video dataset index for the temporal heat map'
    )
    parser.add_argument(
        "--test_idx",
        default=None,
        type=int,
        help='Test video dataset index for the temporal heat map'
    )
    parser.add_argument(
        "--seed",
        default=0,
        type=int,
        help='seed for np.random'
    )
    return parser


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
                        input[i] = input[i].to(device)
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
    print('embeddings size', embeddings.size())
    return embeddings
 

def get_distance_matrix(embeddings, dist_metric):
    embeddings = embeddings
    
    #print('Dist metric:', dist_metric)
    assert(dist_metric in ['cosine', 'euclidean'])
    if dist_metric == 'cosine':
        distance_matrix = cosine_distances(embeddings)
    elif dist_metric == 'euclidean':
        distance_matrix = euclidean_distances(embeddings)
    #print('Distance matrix shape:', distance_matrix.shape)

    np.fill_diagonal(distance_matrix, float('inf'))
    return distance_matrix


def get_closest_data_mat(distance_matrix, top_k):
    idx = np.argpartition(distance_matrix, top_k, axis=-1)
    distance_matrix_topk_unsorted = np.take_along_axis(distance_matrix, idx[:,:top_k], axis=-1)
    idx_sorted_indices = np.argsort(distance_matrix_topk_unsorted, axis=-1)
    top_k = np.take_along_axis(idx, idx_sorted_indices, axis=-1)
    return top_k  # dim: distance_matrix.shape[0] x top_k


def get_closest_data(distance_matrix, exemplar_idx, top_k):
    test_array = distance_matrix[exemplar_idx]
    idx = np.argpartition(test_array, top_k)
    top_k = idx[np.argsort(test_array[idx[:top_k]])]
    return top_k


def plot_img(cfg, fig, data, num_exemplar, row, exemplar_idx, k_idx, spatial_transform=None, temporal_transform=None, output=None):
    exemplar_frame = data._loading_img_path(exemplar_idx, temporal_transform)
    test_frame = [data._loading_img_path(i, temporal_transform) for i in k_idx]

    exemplar_title = '-'.join(exemplar_frame.split('/')[-3:-2])

    print(exemplar_frame)
    print('top k ids:', end=' ')
    for i in k_idx:
        print(i, end=' ')
    print()
    pprint.pprint(test_frame)

    ax = fig.add_subplot(num_exemplar,len(test_frame)+1, row*(len(test_frame)+1)+1)
    image = plt.imread(exemplar_frame)
    plt.imshow(image)
    ax.set_title(exemplar_title, fontsize=5, pad=0.3)
    plt.axis('off')
    for i in range(len(test_frame)):
        test_title = '-'.join(test_frame[i].split('/')[-3:-2])
        ax = fig.add_subplot(num_exemplar,len(test_frame)+1, row*(len(test_frame)+1)+i+2)
        image = plt.imread(test_frame[i])
        plt.imshow(image)
        ax.set_title(test_title, fontsize=5, pad=0.3)
        plt.axis('off')

    with open(os.path.join(output, 'results.txt'), 'a') as f:
        f.write('exemplar_frame:\n{}\n'.format(exemplar_frame))
        for frame in test_frame:
            f.write(frame)
            f.write('\n')
        f.write('\n')

    with open(os.path.join(output, 'exemplar.txt'), 'a') as f:
        f.write('{}, {}'.format(exemplar_idx, exemplar_frame))
        f.write('\n')


def load_exemplar(exemplar_file):
    with open(exemplar_file, 'r') as f:
        lines = f.readlines()
    exemplar_idx  = []
    for line in lines:
        exemplar_idx.append(int(line.split(',')[0].strip()))
    return exemplar_idx


def k_nearest_embeddings(model, test_loader, data, cfg, evaluate_output, num_exemplar, service=None):
    embeddings = evaluate(model, test_loader, log_interval=log_interval)

    distance_matrix = get_distance_matrix(embeddings, cfg.LOSS.DIST_METRIC)

    spatial_transform = build_spatial_transformation(cfg, split)
    temporal_transform = [TemporalCenterFrame()]
    temporal_transform = TemporalCompose(temporal_transform)

    if exemplar_file:
        exemplar_indices = load_exemplar(exemplar_file)
        num_exemplar = len(exemplar_indices)
        print('exemplar_idx retrieved: {}'.format(exemplar_indices))
        print('number of exemplars is: {}'.format(num_exemplar))

    fig = plt.figure()
    for i in range(num_exemplar):
        if not exemplar_file:
            exemplar_idx = np.random.randint(0, distance_matrix.shape[0]-1)
        else:
            exemplar_idx = exemplar_indices[i]

        print('exemplar video id: {}'.format(exemplar_idx))
        k_idx = get_closest_data(distance_matrix, exemplar_idx, top_k)
        k_nearest_data = [data[i] for i in k_idx]
        plot_img(cfg, fig, data, num_exemplar, i, exemplar_idx, k_idx, spatial_transform, temporal_transform, output=evaluate_output)
    # plt.show()
    png_file = os.path.join(evaluate_output, '{}_plot.png'.format(os.path.basename(evaluate_output)))
    fig.tight_layout(pad=3.5)
    plt.savefig(png_file, dpi=300)
    service.upload_file_to_gdrive(png_file, 'evaluate')
    print('figure saved to: {}, and uploaded to GoogleDrive'.format(png_file))


def temporal_heat_map(model, data, cfg, evaluate_output, exemplar_idx=455,
        test_idx=456):

    num_frames_exemplar = data.data[exemplar_idx]['num_frames']

    exemplar_video_full, _, _ = data._get_video_custom_temporal(exemplar_idx)  # full size
    exemplar_video_full = exemplar_video_full.unsqueeze(0)

    num_frames_crop = cfg.DATA.SAMPLE_DURATION
    stride = num_frames_crop // 2
    dists = []

    model.eval()
    with torch.no_grad():
        test_video, _, _ = data.__getitem__(test_idx)  # cropped size
        test_video = test_video.unsqueeze(0)
        print('Test input size:', test_video.size(), '\n')
        if (cfg.MODEL.ARCH == 'slowfast'):
            test_video_in = multipathway_input(test_video, cfg)
            if cuda:
                for i in range(len(test_video_in)):
                    test_video_in[i] = test_video_in[i].to(device)
        else:
            if cuda:
                test_video_in = test_video_in.to(device)
        test_embedding = model(test_video_in)
        #print('Test embed size:', test_embedding.size())

        # Loop across exemplar video, use [i-cfg.DATA.SAMPLE_SIZE,...,i] as the frames for the temporal crop
        for i in range(num_frames_crop, num_frames_exemplar, stride):
            temporal_transform_exemplar = TemporalSpecificCrop(begin_index=i-num_frames_crop, size=num_frames_crop)
            exemplar_video, _, _ = data._get_video_custom_temporal(exemplar_idx, temporal_transform_exemplar)  # full siz
            exemplar_video = exemplar_video.unsqueeze(0)

            if (cfg.MODEL.ARCH == 'slowfast'):
                exemplar_video_in = multipathway_input(exemplar_video, cfg)
                if cuda:
                    for j in range(len(exemplar_video_in)):
                        exemplar_video_in[j] = exemplar_video_in[j].to(device)
            else:
                if cuda:
                    exemplar_video_in = exemplar_video_in.to(device)

            exemplar_embedding = model(exemplar_video_in)
            #print('Exemplar input size:', exemplar_video.size())
            #print('Exemplar embed size:', exemplar_embedding.size())
            dist = F.pairwise_distance(exemplar_embedding, test_embedding, 2)
            dists.append(dist.item())

    #print(dists)
    x = []
    y = []
    plt.show()
    axes = plt.gca()
    axes.set_xlim(0, num_frames_exemplar)
    axes.set_ylim(0, max(dists))
    line, = axes.plot(x, y, 'b-')
    dist_idx = 0

    # channels x frames x width, height --> frames x width x height x channels
    video_ex = exemplar_video_full[0].permute(1,2,3,0)
    video_test = test_video[0].permute(1,2,3,0)
    fps = 25.0
    for i in range(len(video_ex)):
        blank_divider = np.full((2,128,3), 256, dtype=int)
        np_vertical_stack = np.vstack(( video_ex[i].numpy(), blank_divider, video_test[i % len(video_test)].numpy() ))
        cv2.imshow('Videos', np_vertical_stack)

        # show plot of embedding distance for past num_frames_crop frames of exemplar video
        if i >= num_frames_crop and (i - num_frames_crop) % stride == 0:
            x.append(i)
            y.append(dists[dist_idx])
            line.set_xdata(x)
            line.set_ydata(y)
            plt.draw()
            plt.pause(1e-17)
            dist_idx += 1

        cv2.waitKey(int(1.0/fps*1000.0))


if __name__ == '__main__':
    args = m_arg_parser(arg_parser()).parse_args()
    cfg = load_config(args)

    np.random.seed(args.seed)

    force_data_parallel = True
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    global cuda; cuda = torch.cuda.is_available()
    global device; device = torch.device("cuda" if cuda else "cpu")

    name = args.name
    num_exemplar = args.num_exemplar

    if not name:
        name = input('Please specify the name (e.g. ResNet18_K, SlowFast_U): ')
    if not num_exemplar:
        num_exemplar = int(input('Please specify number of exemplar videos: '))

    if not args.output:
        output = input('Please specify output directory: ')
    else:
        output = args.output

    start = time.time()
    now = datetime.now()
    evaluate_output = os.path.join(output, '{}_evaluate'.format(name))
    if not os.path.exists(evaluate_output):
        os.makedirs(evaluate_output)
        print('made output dir:{}'.format(evaluate_output))

    # ============================== Model Setup ===============================

    model=model_selector(cfg)
    print('=> finished generating {} backbone model...'.format(cfg.MODEL.ARCH))

    tripletnet = Tripletnet(model, cfg.LOSS.DIST_METRIC)
    if cuda:
        cfg.NUM_GPUS = torch.cuda.device_count()
        print("Using {} GPU(s)".format(cfg.NUM_GPUS))
        if cfg.NUM_GPUS > 1 or force_data_parallel:
            tripletnet = nn.DataParallel(tripletnet)

    if args.checkpoint_path is not None:
        start_epoch, best_acc = load_checkpoint(tripletnet, args.checkpoint_path)

    model = tripletnet.module.embeddingnet
    if cuda:
        model.to(device)

    print('=> finished generating similarity network...')

    # ============================== Data Loaders ==============================

    test_loader, data = data_loader.build_data_loader(split, cfg, triplets=False)
    print()

    # ================================ Evaluate ================================

    if args.heatmap:
        if args.ex_idx and args.test_idx:
            temporal_heat_map(model, data, cfg, evaluate_output, args.ex_idx,
                args.test_idx)
        else:
            print ('No exemplar and test indices provided')
            temporal_heat_map(model, data, cfg, evaluate_output)
    else:
        k_nearest_embeddings(model, test_loader, data, cfg, evaluate_output,
                num_exemplar, service=GoogleDriveUploader())
        print('total runtime: {}s'.format(time.time()-start))
