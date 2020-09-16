"""
Created by Sherry Chen on Jul 3, 2020
Build and Train Triplet network. Supports saving and loading checkpoints,
"""

import sys, os
#import gc
import numpy as np
import time
import argparse
import tqdm
import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from models.triplet_net import Tripletnet
from datasets import data_loader
from models.model_utils import (model_selector, multipathway_input,
                            load_pretrained_model, save_checkpoint, load_checkpoint,
                            AverageMeter, accuracy, create_output_dirs)
from config.m_parser import load_config, arg_parser
import misc.distributed_helper as du_helper
from datasets.loss import OnlineTripleLoss
from train import validate

log_interval = 5 #log interval for batch number


def train_epoch(train_loader, model, criterion, optimizer, epoch, cfg, cuda, device, is_master_proc=True):
    losses = AverageMeter()
    accs = AverageMeter()

    running_n_triplets = 0
    running_loss = 0

    # switching to training mode
    model.train()
    start = time.time()

    world_size = du_helper.get_world_size()
    sampling_strategy = 'random_semi_hard'

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # if batch_idx > 2:
        #     break
        anchor, positive = inputs
        (a_target, p_target) = targets
        batch_size = anchor.size(0)
        targets = torch.cat((a_target, p_target), 0)

        if cfg.MODEL.ARCH == 'slowfast':
            anchor = multipathway_input(anchor, cfg)
            positive = multipathway_input(positive, cfg)
            if cuda:
                for i in range(len(anchor)):
                    anchor[i], positive[i]= anchor[i].to(device), positive[i].to(device)
        elif cuda:
            anchor, positive = anchor.to(device), positive.to(device)

        anchor_outputs = model(anchor)
        positive_outputs = model(positive)
        outputs = torch.cat((anchor_outputs, positive_outputs), 0)
        if cuda:
            targets = targets.to(device)

        loss, n_triplets = criterion(outputs, targets, sampling_strategy=sampling_strategy)

        # # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (cfg.NUM_GPUS > 1):
            loss = du_helper.all_reduce([loss], avg=True)
            loss = loss[0]

        batch_size_world = batch_size * world_size

        losses.update(loss.item(), batch_size_world)
        running_n_triplets += n_triplets

        if (is_master_proc) and (batch_idx + 1) % log_interval == 0:
            print(
                f"Training: {epoch}, {batch_idx+1}\
                Loss:{round(losses.avg, 4)}\
                N_Triplets:{running_n_triplets/log_interval}"
            )
            running_n_triplets = 0

    if (is_master_proc):
        print('\nTrain set: Average loss: {:.4f}%\n'.format(losses.avg))
        print('epoch:{} runtime:{}'.format(epoch, (time.time()-start)/3600))
        with open('{}/tnet_checkpoints/train_loss_and_acc.txt'.format(cfg.OUTPUT_PATH), "a") as f:
            f.write('epoch:{} runtime:{} {:.4f}\n'.format(epoch, round((time.time()-start)/3600,2), losses.avg))
        print('saved to file:{}'.format('{}/tnet_checkpoints/train_loss_and_acc.txt'.format(cfg.OUTPUT_PATH)))


def train(args, cfg):
    best_acc = 0
    start_epoch = 0
    cudnn.benchmark = True

    cuda = torch.cuda.is_available()
    device = torch.cuda.current_device()

    is_master_proc = du_helper.is_master_proc(cfg.NUM_GPUS)

    if is_master_proc:
        create_output_dirs(cfg)

    # ======================== Similarity Network Setup ========================
    model=model_selector(cfg)
    if(is_master_proc):
        print('\n=> finished generating {} backbone model...'.format(cfg.MODEL.ARCH))

    # Load pretrained backbone if path exists
    if args.pretrain_path is not None:
        model = load_pretrained_model(model, args.pretrain_path, is_master_proc)

    if cuda:
        model = model.cuda(device=device)
        if torch.cuda.device_count() > 1:
            #model = nn.DataParallel(model)
            if cfg.MODEL.ARCH == '3dresnet':
                model = torch.nn.parallel.DistributedDataParallel(module=model, device_ids=[device], find_unused_parameters=True, broadcast_buffers=False)
            else:
                model = torch.nn.parallel.DistributedDataParallel(module=model, device_ids=[device], broadcast_buffers=False)

    # Load similarity network checkpoint if path exists
    if args.checkpoint_path is not None:
        start_epoch, best_acc = load_checkpoint(model, args.checkpoint_path, is_master_proc)

    # Triplet net used for validation
    tripletnet = Tripletnet(model, cfg.LOSS.DIST_METRIC)
    if cuda:
        tripletnet = tripletnet.cuda(device=device)
        if torch.cuda.device_count() > 1:
            if cfg.MODEL.ARCH == '3dresnet':
                tripletnet = torch.nn.parallel.DistributedDataParallel(module=tripletnet, device_ids=[device], find_unused_parameters=True)
            else:
                tripletnet = torch.nn.parallel.DistributedDataParallel(module=tripletnet, device_ids=[device])

    if(is_master_proc):
        print('=> finished generating similarity network...')

    # ============================== Data Loaders ==============================
    train_loader, (_, train_sampler) = data_loader.build_data_loader('train', cfg, is_master_proc, triplets=True)
    val_loader, _ = data_loader.build_data_loader('val', cfg, is_master_proc, triplets=True, negative_sampling=True)

    # # ======================== Loss and Optimizer Setup ========================
    val_criterion = torch.nn.MarginRankingLoss(margin=cfg.LOSS.MARGIN).to(device)
    criterion = OnlineTripleLoss(margin=cfg.LOSS.MARGIN, dist_metric=cfg.LOSS.DIST_METRIC).to(device)
    optimizer = optim.SGD(model.parameters(), lr=cfg.OPTIM.LR, momentum=cfg.OPTIM.MOMENTUM)

    print('\n==> using criterion:{} for training task'.format(criterion))
    print('==> using criterion:{} for validation task'.format(val_criterion))

    n_parameters = sum([p.data.nelement() for p in model.parameters()])
    if(is_master_proc):
        print('\n + Number of params: {}'.format(n_parameters))

    # # ============================= Training loop ==============================
    for epoch in range(start_epoch, cfg.TRAIN.EPOCHS):
        if (is_master_proc):
            print ('\nEpoch {}/{}'.format(epoch, cfg.TRAIN.EPOCHS-1))
        if cfg.NUM_GPUS > 1:
            train_sampler.set_epoch(epoch)
        train_epoch(train_loader, model, criterion, optimizer, epoch, cfg, cuda, device, is_master_proc)
        acc = validate(val_loader, tripletnet, val_criterion, epoch, cfg, cuda, device, is_master_proc)
        is_best = acc > best_acc
        best_acc = max(acc, best_acc)
        save_checkpoint({
            'epoch': epoch+1,
            'state_dict':model.state_dict(),
            'best_prec1': best_acc,
        }, is_best, cfg.MODEL.ARCH, cfg.OUTPUT_PATH, is_master_proc)


if __name__ == '__main__':
    args = arg_parser().parse_args()
    cfg = load_config(args)

    shard_id = args.shard_id
    if args.compute_canada:
        print('running on compute canada')
        shard_id = int(os.environ['SLURM_NODEID'])

    print ('Total nodes:', args.num_shards)
    print ('Node id:', shard_id)

    # Set visible gpu devices
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

    if torch.cuda.is_available():
        cfg.NUM_GPUS = torch.cuda.device_count()
        print("Using {} GPU(s) per node".format(cfg.NUM_GPUS))

    # Launch processes for all gpus
    du_helper.launch_processes(args, cfg, func=train, shard_id=shard_id, NUM_SHARDS=args.num_shards, ip_address_port=args.ip_address_port)
