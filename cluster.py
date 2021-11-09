"""
Created by Sherry Chen on Jul 3, 2020
Build and Train Triplet network. Supports saving and loading checkpoints,
"""

#def warn(*args, **kwargs):
#        pass
#import warnings
#warnings.warn = warn

import sys, os
from networkx.algorithms import cluster
#import gc
import numpy as np
import time
import argparse
import tqdm
import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from validation import validate
from evaluate import k_nearest_embeddings, get_embeddings_and_labels
from models.triplet_net import Tripletnet
from datasets import data_loader
from models.model_utils import (model_selector, multipathway_input,
                            load_pretrained_model, save_checkpoint, load_checkpoint,
                            AverageMeter, accuracy, create_output_dirs)
from config.m_parser import load_config, arg_parser
import misc.distributed_helper as du_helper
from loss.triplet_loss import OnlineTripletLoss, MemTripletLoss
from loss.NCE_loss import NCEAverage, NCEAverage_intra_neg, NCESoftmaxLoss
from clustering.cluster_masks import fit_cluster
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score

#TODO: add this to config file
modality = 'rgb'
intra_neg = False #True
moco = False #True
neg_type='repeat'

import cv2

# Setup training and run training loop
def train(args, cfg):

    #torch.autograd.set_detect_anomaly(True)
    cudnn.benchmark = True
    # Cuda and current device
    cuda = torch.cuda.is_available()
    device = torch.cuda.current_device()

    # Load similarity network checkpoint if path exists
    # cfg.DATASET.CLUSTER_PATH = '{}/vid_clusters.txt'.format(cfg.OUTPUT_PATH)
    # if(is_master_proc):
    #     print('\n==> Building validation data loader (triplet)...')
    # val_loader, (_, _) = data_loader.build_data_loader('val', cfg, is_master_proc, triplets=True, negative_sampling=True)

    # # Setting is_master_proc to false when loading single video data loaders 
    # # deliberately to not re-print data loader information

    # if(is_master_proc):
    print('\n==> Building training data loader (single video)...')
    eval_train_loader, (train_data, _) = data_loader.build_data_loader('train',
            cfg, True, triplets=False, req_train_shuffle=False,
            drop_last=False)

    # if(is_master_proc):
    #     print('\n==> Building validation data loader (single video)...')
    # eval_val_loader, (val_data, _) = data_loader.build_data_loader('val', cfg,
    #         is_master_proc, triplets=False, val_sample=None,
    #         drop_last=False, batch_size=1)

    # ============================= Training loop ==============================

    # embeddings_computed = True

    # if is_master_proc or not embeddings_computed:
    start_time = time.time()
    cfg.OUTPUT_PATH = '/home/sherry/output/finch/finch/resnet_ucf_ic_llc_optical_pos_replace_0.2'
    embeddings, true_labels, idxs = get_embeddings_and_labels(args, cfg,
            None, cuda, device, None, split='train',
            load_pkl=True, save_pkl=False)
    print('Time to get embeddings: {:.2f}s'.format(time.time()-start_time))

    # Cluster
    print('\n=> Clustering')
    start_time = time.time()
    print('embeddings shape', embeddings.size())

    cluster_labels = fit_cluster(embeddings, cfg.ITERCLUSTER.METHOD,
                            cfg.ITERCLUSTER.K, cfg.ITERCLUSTER.L2_NORMALIZE,
                            -1 )
                            #cfg.ITERCLUSTER.FINCH_PARTITION)

    # cluster_labels = cluster_labels[:,0]
    # print('Time to cluster: {:.2f}s'.format(time.time()-start_time))


    classind = '/media/diskstation/datasets/UCF101/data/ucf101/ClassInd.txt'
    with open(classind, 'r') as f:
        classind = f.readlines() 
    
    classind = [classname.replace('\n', '') for classname in classind]
    print(classind)
    # print([label for label in true_labels if label == 0])
    for partition in range(0,6):
        cur_cluster_label = cluster_labels[:,partition]
        res = {}
        for i in range(len(cur_cluster_label)):
            label = cur_cluster_label[i]
            true_label = true_labels[i]
            true_name = classind[true_label]
            if str(label) not in res:
                res[str(label)] = []
            res[str(label)].append(true_name)
        for cluster in res:
            res[cluster].sort()
        
        import json
        with open('finch_partition_{}.json'.format(partition), 'w') as f:
            json.dump(res, f)


    # # Calculate NMI for true labels vs cluster assignments
    # #true_labels = train_data.get_total_labels()
    # NMI = normalized_mutual_info_score(true_labels, cluster_labels)
    # print('NMI between true labels and cluster assignments: {:.3f}'.format(NMI))

    # # Calculate Adjusted NMI for true labels vs cluster assignements
    # AMI = adjusted_mutual_info_score(true_labels, cluster_labels)
    # print('AMI between true labels and cluster assignments: {:.3f}\n'.format(AMI))


    # # Update probability of sampling positive from same video using NMI
    # if cfg.ITERCLUSTER.ADAPTIVEP:
    #     cfg.DATASET.POSITIVE_SAMPLING_P = float(1.0 - NMI)

    # # Get cluster assignments in unshuffled order of dataset
    # cluster_assignments_unshuffled_order = [None] * len(eval_train_loader.dataset)
    # for i in range(len(cluster_labels)):
    #     cluster_assignments_unshuffled_order[idxs[i]] = cluster_labels[i]

    # # Save cluster assignments corresponding to unshuffled order of dataset
    # cluster_output_path = os.path.join(cfg.OUTPUT_PATH, 'vid_clusters.txt')
    # with open(cluster_output_path, "w") as f:
    #     for label in cluster_assignments_unshuffled_order:
    #         f.write('{}\n'.format(label))
    # print('Saved cluster labels to', cluster_output_path)



if __name__ == '__main__':

    torch.manual_seed(7)
    np.random.seed(7)
    torch.cuda.manual_seed_all(7)

    print ('\n==> Parsing parameters:')
    args = arg_parser().parse_args()
    cfg = load_config(args)

    if args.vector:
        assert args.checkpoint_path is not None

    # If iteratively clustering, overwrite the cluster_path
    print('Iteratively clustering?: {}, warmup epochs = {}'.format(args.iterative_cluster,
        cfg.ITERCLUSTER.WARMUP_EPOCHS))
    print('Relative speed perception loss?:', cfg.LOSS.RELATIVE_SPEED_PERCEPTION)
    print('local local contrast loss?:', cfg.LOSS.LOCAL_LOCAL_CONTRAST)
    if args.iterative_cluster:
        assert(cfg.DATASET.TARGET_TYPE_T == 'cluster_label' and cfg.DATASET.POSITIVE_SAMPLING_P != 1.0)

    print('Multiview positives ({}% chance replace): {}'.format(cfg.DATASET.PROB_POS_CHANNEL_REPLACE*100,
        cfg.DATASET.POS_CHANNEL_REPLACE))
    print('Spatio-temporal-attention?: {}'.format(cfg.RESNET.ATTENTION))

    # Set shard_id to $SLURM_NODEID if running on compute canada
    shard_id = args.shard_id
    if args.compute_canada:
        print('Running on compute canada')
        shard_id = int(os.environ['SLURM_NODEID'])

    # Print node information
    print ('Total nodes:', args.num_shards)
    print ('Node id:', shard_id)

    # Set visible GPU devices and print gpu information
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    if torch.cuda.is_available():
        cfg.NUM_GPUS = torch.cuda.device_count()
        print("Using {} GPU(s) per node".format(cfg.NUM_GPUS))

    # Print training parameters
    print('Triplet sampling strategy: {}'.format(cfg.DATASET.SAMPLING_STRATEGY))
    print('Probability of sampling positive from same video: {}'.format(cfg.DATASET.POSITIVE_SAMPLING_P))
    print('OUTPUT_PATH is set to: {}'.format(cfg.OUTPUT_PATH))
    print('BATCH_SIZE is set to: {}'.format(cfg.TRAIN.BATCH_SIZE))
    print('NUM_WORKERS is set to: {}'.format(cfg.TRAIN.NUM_DATA_WORKERS))
    print('SAMPLE SIZE is set to: {}'.format(cfg.DATA.SAMPLE_SIZE))
    print('N_CLASSES is set to: {}'.format(cfg.RESNET.N_CLASSES))
    print('ITERCLUSTER.INTERVAL is set to: {}'.format(cfg.ITERCLUSTER.INTERVAL))
    print('ITERCLUSTER.ADAPTIVEP is set to: {}'.format(cfg.ITERCLUSTER.ADAPTIVEP))
    print('Learning rate is set to {}'.format(cfg.OPTIM.LR))
    print('Momentum set to {}'.format(cfg.OPTIM.MOMENTUM))
    print('Margin set to {}'.format(cfg.LOSS.MARGIN))

    # Launch processes for all gpus
    print('\n==> Launching gpu processes...')
    du_helper.launch_processes(args, cfg, func=train, shard_id=shard_id,
        NUM_SHARDS=args.num_shards, ip_address_port=args.ip_address_port)
