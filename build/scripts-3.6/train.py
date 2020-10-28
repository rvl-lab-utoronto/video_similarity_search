"""
Created by Sherry Chen on Jul 3, 2020
Build and Train Triplet network. Supports saving and loading checkpoints,
"""

import sys, os
#import gc
import time
# import csv
import argparse
import tqdm
import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from validation import validate
from models.triplet_net import Tripletnet
from datasets import data_loader
from models.model_utils import (model_selector, multipathway_input,
                            load_pretrained_model, save_checkpoint, load_checkpoint,
                            AverageMeter, accuracy, create_output_dirs)

from config.m_parser import load_config, arg_parser
import misc.distributed_helper as du_helper
import evaluate

def train_epoch(train_loader, tripletnet, criterion, optimizer, epoch, cfg, cuda, device, is_master_proc=True):
    losses = AverageMeter()
    accs = AverageMeter()
    emb_norms = AverageMeter()
    # switching to training mode
    tripletnet.train()
    start = time.time()

    world_size = du_helper.get_world_size()

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        anchor, positive, negative = inputs
        batch_size = anchor.size(0)

        if cfg.MODEL.ARCH == 'slowfast':
            anchor = multipathway_input(anchor, cfg)
            positive = multipathway_input(positive, cfg)
            negative = multipathway_input(negative, cfg)
            if cuda:
                for i in range(len(anchor)):
                    anchor[i], positive[i], negative[i] = anchor[i].to(device), positive[i].to(device), negative[i].to(device)
        else:
            if cuda:
                anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

        dista, distb, embedded_x, embedded_y, embedded_z = tripletnet(anchor, positive, negative)

        #1 means, dista should be larger than distb
        target = torch.FloatTensor(dista.size()).fill_(-1)
        if cuda:
            target = target.to(device)

        loss = criterion(dista, distb, target)
        if loss.size() == 1 and loss == 0:
            continue
        embedd_norm_sum = embedded_x.norm(2) + embedded_y.norm(2) + embedded_z.norm(2)

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy
        acc = accuracy(dista.detach(), distb.detach())
        # Gather the predictions across all the devices (sum)
        if (cfg.NUM_GPUS > 1):
            loss, acc, embedd_norm_sum = du_helper.all_reduce([loss, acc, embedd_norm_sum], avg=True)

        batch_size_world = batch_size * world_size

        # record loss and accuracy
        losses.update(loss.item(), batch_size_world)
        accs.update(acc.item(), batch_size_world)

        emb_norms.update(embedd_norm_sum.item()/3, batch_size_world)

        if ((batch_idx + 1) * world_size) % cfg.TRAIN.LOG_INTERVAL == 0:
            if (is_master_proc):
                print('Train Epoch: {} [{}/{} | {:.1f}%]\t'
                    'Loss: {:.4f} ({:.4f}) \t'
                    'Acc: {:.2f}% ({:.2f}%) \t'
                    'Emb_Norm: {:.2f} ({:.2f})'.format(
                    epoch, (batch_idx + 1) * batch_size_world,
                    len(train_loader.dataset), 100. * ((batch_idx + 1) *
                        batch_size_world / len(train_loader.dataset)),
                    losses.val, losses.avg,
                    100. * accs.val, 100. * accs.avg, emb_norms.val, emb_norms.avg))

    if (is_master_proc):
        print('\nTrain set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
            losses.avg, 100. * accs.avg))

        print('Epoch:{} runtime:{}'.format(epoch, (time.time()-start)/3600))
        with open('{}/tnet_checkpoints/train_loss_and_acc.txt'.format(cfg.OUTPUT_PATH), "a") as f:
            f.write('epoch:{} runtime:{} {:.4f} {:.2f}\n'.format(epoch, round((time.time()-start)/3600,2), losses.avg, 100. * accs.avg))
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

    tripletnet = Tripletnet(model, cfg.LOSS.DIST_METRIC)

    if cuda:
        tripletnet = tripletnet.cuda(device=device)
        if torch.cuda.device_count() > 1:
            #tripletnet = nn.DataParallel(tripletnet)
            if cfg.MODEL.ARCH == '3dresnet':
                tripletnet = torch.nn.parallel.DistributedDataParallel(module=tripletnet, device_ids=[device], find_unused_parameters=True)
            else:
                tripletnet = torch.nn.parallel.DistributedDataParallel(module=tripletnet, device_ids=[device])

    # Load similarity network checkpoint if path exists
    if args.checkpoint_path is not None:
        start_epoch, best_acc = load_checkpoint(tripletnet, args.checkpoint_path, is_master_proc)

    if(is_master_proc):
        print('=> finished generating similarity network...')

    # ============================== Data Loaders ==============================

    train_loader, _ = data_loader.build_data_loader('train', cfg, is_master_proc, triplets=True, negative_sampling=True)
    val_loader, _ = data_loader.build_data_loader('val', cfg, is_master_proc, triplets=True, negative_sampling=True)

    # ======================== Loss and Optimizer Setup ========================

    criterion = torch.nn.MarginRankingLoss(margin=cfg.LOSS.MARGIN).to(device)
    optimizer = optim.SGD(tripletnet.parameters(), lr=cfg.OPTIM.LR, momentum=cfg.OPTIM.MOMENTUM)

    n_parameters = sum([p.data.nelement() for p in tripletnet.parameters()])
    if(is_master_proc):
        print('\n + Number of params: {}'.format(n_parameters))

    # ============================= Training loop ==============================

    for epoch in range(start_epoch, cfg.TRAIN.EPOCHS):
        if(is_master_proc):
            print ('\nEpoch {}/{}'.format(epoch, cfg.TRAIN.EPOCHS-1))
        train_epoch(train_loader, tripletnet, criterion, optimizer, epoch, cfg, cuda, device, is_master_proc)
        acc = validate(val_loader, tripletnet, criterion, epoch, cfg, cuda, device, is_master_proc)
        is_best = acc > best_acc
        best_acc = max(acc, best_acc)
        save_checkpoint({
            'epoch': epoch+1,
            'state_dict':tripletnet.state_dict(),
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