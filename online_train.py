"""
Created by Sherry Chen on Jul 3, 2020
Build and Train Triplet network. Supports saving and loading checkpoints,
"""

import sys, os
#import gc
import numpy as np
import time
import csv
import argparse
import shutil
import tqdm
import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from models.triplet_net import Tripletnet
from datasets import data_loader
from models.model_utils import model_selector, multipathway_input
from config.m_parser import load_config, arg_parser
import misc.distributed_helper as du_helper
from datasets.loss import OnlineTripleLoss

log_interval = 5 #log interval for batch number

def load_pretrained_model(model, pretrain_path, is_master_proc=True):
    if pretrain_path:
        if (is_master_proc):
            print('loading pretrained model {}'.format(pretrain_path))
        pretrain = torch.load(pretrain_path, map_location='cpu')
        model.load_state_dict(pretrain['state_dict'])
    return model


def save_checkpoint(state, is_best, model_name, output_path, is_master_proc=True, filename='checkpoint.pth.tar'):
    # Save checkpoints only from the master process
    if not is_master_proc:
        return

    """Saves checkpoint to disk"""
    directory = "tnet_checkpoints/%s/"%(model_name)
    directory = os.path.join(output_path, directory)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if (is_master_proc):
        print('=> checkpoint:{} saved...'.format(filename))
    if is_best:
        shutil.copyfile(filename,  os.path.join(directory, 'model_best.pth.tar'))
        if (is_master_proc):
            print('=> best_model saved as:{}'.format(os.path.join(directory, 'model_best.pth.tar')))


def load_checkpoint(model, checkpoint_path, is_master_proc=True):
    if os.path.isfile(checkpoint_path):
        if (is_master_proc):
            print("=> loading checkpoint '{}'".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        if (is_master_proc):
            print("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_path, checkpoint['epoch']))
    else:
        if (is_master_proc):
            print("=> no checkpoint found at '{}'".format(checkpoint_path))

    return start_epoch, best_prec1


def train_epoch(train_loader, model, criterion, optimizer, epoch, cfg, is_master_proc=True, p=1.0):
    losses = AverageMeter()
    accs = AverageMeter()
    # emb_norms = AverageMeter()

    running_n_triplets = 0
    running_loss = 0
    # switching to training mode
    model.train()
    start = time.time()

    world_size = du_helper.get_world_size()

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # if batch_idx > 2:
        #     break

        anchor, positive, _ = inputs
        batch_size = anchor.size(0)
        # print('batch_size', batch_size)
        if cuda:
            anchor = anchor.to(device)
            positive = positive.to(device)
        anchor_outputs = model(anchor)
        positive_outputs = model(positive)

        outputs = torch.cat((anchor_outputs, positive_outputs), 0)

        # #1 means, dista should be larger than distb
        # target = torch.FloatTensor(dista.size()).fill_(-1)
        if cuda:
            targets = torch.cat(targets[:2], 0)
            print(targets)
            targets = targets.to(device)

        if np.random.random_sample() < p:
            sampling_strategy = 'random_negative'
            anchor_target = torch.tensor(range(0, anchor.size(0)), dtype=torch.int)
            positive_target = torch.tensor(range(0, positive.size(0)), dtype=torch.int)
            targets = torch.cat((anchor_target, positive_target), 0)
            print('targets', targets)

        else:
            sampling_strategy = 'random_semi_hard'

        loss, n_triplets = criterion(outputs, targets, sampling_strategy=sampling_strategy)
        # embedd_norm_sum = embedded_x.norm(2) + embedded_y.norm(2) + embedded_z.norm(2)

        # # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (cfg.NUM_GPUS > 1):
            loss = du_helper.all_reduce([loss], avg=True)

        batch_size_world = batch_size * world_size

        losses.update(loss.item(), batch_size_world)
        running_n_triplets += n_triplets
        # running_loss += loss.item() if n_triplets > 0 else 0
        # losses.

        if (is_master_proc) and (batch_idx + 1) % log_interval == 0:
            print(
                f"Training: {epoch}, {batch_idx+1}\
                Loss:{round(losses.avg, 4)}\
                N_Triplets:{running_n_triplets/log_interval}"
            )
            # running_loss = 0.0
            running_n_triplets = 0

    if (is_master_proc):
        print('\nTrain set: Average loss: {:.4f}%\n'.format(losses.avg))
        print('epoch:{} runtime:{}'.format(epoch, (time.time()-start)/3600))
        with open('{}/tnet_checkpoints/train_loss_and_acc.txt'.format(cfg.OUTPUT_PATH), "a") as f:
            f.write('epoch:{} runtime:{} {:.4f}\n'.format(epoch, round((time.time()-start)/3600,2), losses.avg))
        print('saved to file:{}'.format('{}/tnet_checkpoints/train_loss_and_acc.txt'.format(cfg.OUTPUT_PATH)))




def validate(val_loader, model, criterion, epoch, cfg, is_master_proc=True):
    losses = AverageMeter()
    accs = AverageMeter()

    world_size = du_helper.get_world_size()

    tripletnet = Tripletnet(model, cfg.LOSS.DIST_METRIC)

    if cuda:
        tripletnet = tripletnet.cuda(device=device)
        if torch.cuda.device_count() > 1:
            #tripletnet = nn.DataParallel(tripletnet)
            if cfg.MODEL.ARCH == '3dresnet':
                tripletnet = torch.nn.parallel.DistributedDataParallel(module=tripletnet, device_ids=[device], find_unused_parameters=True)
            else:
                tripletnet = torch.nn.parallel.DistributedDataParallel(module=tripletnet, device_ids=[device])


    tripletnet.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            (anchor, positive, negative) = inputs
            (anchor_target, positive_target, negative_target) = targets

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

            target = torch.FloatTensor(dista.size()).fill_(-1)
            if cuda:
                target = target.to(device)

            loss = criterion(dista, distb, target)

            # measure accuracy
            acc = accuracy(dista.detach(), distb.detach())

            # record loss and accuracy
            accs.update(acc.item(), batch_size)
            losses.update(loss.item(), batch_size)

    if cfg.NUM_GPUS > 1:
        acc_sum = torch.tensor([accs.sum], dtype=torch.float32, device=device)
        acc_count = torch.tensor([accs.count], dtype=torch.float32, device=device)

        losses_sum = torch.tensor([losses.sum], dtype=torch.float32, device=device)
        losses_count = torch.tensor([losses.count], dtype=torch.float32, device=device)

        acc_sum, losses_sum, acc_count, losses_count = du_helper.all_reduce([acc_sum, losses_sum, acc_count, losses_count], avg=False)

        accs.avg = acc_sum.item() / acc_count.item()
        losses.avg = losses_sum.item() / losses_count.item()

    if (is_master_proc):
        print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
            losses.avg, 100. * accs.avg))

        with open('{}/tnet_checkpoints/val_loss_and_acc.txt'.format(cfg.OUTPUT_PATH), "a") as val_file:
            val_file.write('epoch:{} {:.4f} {:.2f}\n'.format(epoch, losses.avg, 100. * accs.avg))

    return accs.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(dista, distb):
    margin = 0
    pred = (distb - dista - margin)
    return (pred > 0).sum() * 1.0 / (dista.size()[0])


def create_output_dirs(cfg):
    if not os.path.exists(cfg.OUTPUT_PATH):
        os.makedirs(cfg.OUTPUT_PATH)

    if not os.path.exists(os.path.join(cfg.OUTPUT_PATH, 'tnet_checkpoints')):
        os.makedirs(os.path.join(cfg.OUTPUT_PATH, 'tnet_checkpoints'))


def train(args, cfg):
    best_acc = 0
    start_epoch = 0
    cudnn.benchmark = True

    global cuda; cuda = torch.cuda.is_available()
    global device; device = torch.cuda.current_device()

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

    # tripletnet = Tripletnet(model, cfg.LOSS.DIST_METRIC)

    if cuda:
        model = model.cuda(device=device)
        if torch.cuda.device_count() > 1:
            #model = nn.DataParallel(model)
            if cfg.MODEL.ARCH == '3dresnet':
                model = torch.nn.parallel.DistributedDataParallel(module=model, device_ids=[device], find_unused_parameters=True)
            else:
                model = torch.nn.parallel.DistributedDataParallel(module=model, device_ids=[device])

    # Load similarity network checkpoint if path exists
    if args.checkpoint_path is not None:
        start_epoch, best_acc = load_checkpoint(model, args.checkpoint_path, is_master_proc)

    if(is_master_proc):
        print('=> finished generating similarity network...')

    # ============================== Data Loaders ==============================

    train_loader, _ = data_loader.build_data_loader('train', cfg, is_master_proc, triplets=True)
    val_loader, _ = data_loader.build_data_loader('val', cfg, is_master_proc, triplets=True)

    # # ======================== Loss and Optimizer Setup ========================
    #
    val_criterion = torch.nn.MarginRankingLoss(margin=cfg.LOSS.MARGIN).to(device)
    criterion = OnlineTripleLoss(margin=cfg.LOSS.MARGIN).to(device)
    optimizer = optim.SGD(model.parameters(), lr=cfg.OPTIM.LR, momentum=cfg.OPTIM.MOMENTUM)
    #
    n_parameters = sum([p.data.nelement() for p in model.parameters()])
    if(is_master_proc):
        print('\n + Number of params: {}'.format(n_parameters))
    #
    # # ============================= Training loop ==============================
    #

    for epoch in range(start_epoch, cfg.TRAIN.EPOCHS):
        if(is_master_proc):
            print ('\nEpoch {}/{}'.format(epoch, cfg.TRAIN.EPOCHS-1))
        train_epoch(train_loader, model, criterion, optimizer, epoch, cfg, is_master_proc)
        acc = validate(val_loader, model, val_criterion, epoch, cfg, is_master_proc)
        is_best = acc > best_acc
        best_acc = max(acc, best_acc)
        save_checkpoint({
            'epoch': epoch+1,
            'state_dict':model.state_dict(),
            'best_prec1': best_acc,
        }, is_best, cfg.MODEL.ARCH, cfg.OUTPUT_PATH, is_master_proc)
    #

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
