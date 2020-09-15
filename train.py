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

from models.triplet_net import Tripletnet
from datasets import data_loader
from models.model_utils import (model_selector, multipathway_input,
                            load_pretrained_model, save_checkpoint, load_checkpoint,
                            AverageMeter, accuracy, create_output_dirs)

from config.m_parser import load_config, arg_parser
import misc.distributed_helper as du_helper
import evaluate


log_interval = 2 #log interval for batch number

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

        if ((batch_idx + 1) * world_size) % log_interval == 0:
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


def get_topk_acc(embeddings, labels, dist_metric):
    distance_matrix = evaluate.get_distance_matrix(embeddings, dist_metric)
    top1_sum = 0
    top5_sum = 0

    top1_indices = evaluate.get_closest_data_mat(distance_matrix, top_k=1)  # dim: distance_matrix.shape[0] x top_k
    top5_indices = evaluate.get_closest_data_mat(distance_matrix, top_k=5)  # dim: distance_matrix.shape[0] x top_k

    for i, label in enumerate(labels):
        top1_idx = top1_indices[i]
        top5_idx = top5_indices[i]
        top1_label = [labels[j] for j in top1_idx]
        top5_labels = [labels[j] for j in top5_idx]
        print(i, 'cur', label, 'top1', top1_label, 'top5', top5_labels)
        top1_sum += int(label in top1_label) 
        top5_sum += int(label in top5_labels)

    top1_acc = top1_sum / len(labels)
    top5_acc = top5_sum / len(labels)
    return top1_acc, top5_acc


def validate(val_loader, tripletnet, criterion, epoch, cfg, cuda, device, is_master_proc=True):
    losses = AverageMeter()
    accs = AverageMeter()
    top1_accs = AverageMeter()
    top5_accs = AverageMeter()
    embeddings = []
    labels = []

    world_size = du_helper.get_world_size()

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
                anchor_target = anchor_target.to(device)

            # Triplet loss
            loss = criterion(dista, distb, target)

            # measure accuracy
            acc = accuracy(dista.detach(), distb.detach())

            # Top 1/5 Acc - use anchors and positives so there is at least 
            # 1 positive per anchor
            embeddings = torch.cat((embedded_x.detach().cpu(), embedded_y.detach().cpu()), dim=0)
            labels = torch.cat((anchor_target.detach().cpu(), positive_target.detach().cpu()), dim=0)
            top1_acc, top5_acc = get_topk_acc(embeddings, labels.tolist(), cfg.LOSS.DIST_METRIC)

            # record loss and accuracy
            accs.update(acc.item(), batch_size)
            top1_accs.update(top1_acc)
            top5_accs.update(top5_acc)
            losses.update(loss.item(), batch_size)

            batch_size_world = batch_size * world_size
            if ((batch_idx + 1) * world_size) % log_interval == 0:
                if (is_master_proc):
                    print('Val Epoch: {} [{}/{} | {:.1f}%]\t'
                          'Loss: {:.4f} ({:.4f}) \t'
                          'Acc: {:.2f}% ({:.2f}%) \t'
                          'Top1 Acc: {:.2f}% ({:.2f}%) \t'
                          'Top5 Acc: {:.2f}% ({:.2f}%)'.format(
                              epoch, (batch_idx + 1) * batch_size_world, 
                              len(val_loader.dataset), 100. * ((batch_idx + 1) * batch_size_world / len(val_loader.dataset)), 
                              losses.val, losses.avg, 100. * accs.val, 100. * accs.avg, 
                              100. * top1_accs.val, 100. * top1_accs.avg, 100. * top5_accs.val, 100. * top5_accs.avg))

    if cfg.NUM_GPUS > 1:
        acc_sum = torch.tensor([accs.sum], dtype=torch.float32, device=device)
        acc_count = torch.tensor([accs.count], dtype=torch.float32, device=device)

        top1_acc_sum = torch.tensor([top1_accs.sum], dtype=torch.float32, device=device)
        top1_acc_count = torch.tensor([top1_accs.count], dtype=torch.float32, device=device)

        top5_acc_sum = torch.tensor([top5_accs.sum], dtype=torch.float32, device=device)
        top5_acc_count = torch.tensor([top5_accs.count], dtype=torch.float32, device=device)

        losses_sum = torch.tensor([losses.sum], dtype=torch.float32, device=device)
        losses_count = torch.tensor([losses.count], dtype=torch.float32, device=device)

        acc_sum, top1_acc_sum, top5_acc_sum, losses_sum, \
            acc_count, top1_acc_count, top5_acc_count, losses_count = \
            du_helper.all_reduce([acc_sum, top1_acc_sum, top5_acc_sum, losses_sum, acc_count, top1_acc_count, top5_acc_count, losses_count], avg=False)

        accs.avg = acc_sum.item() / acc_count.item()
        top1_accs.avg = top1_acc_sum.item() / top1_acc_count.item()
        top5_accs.avg = top5_acc_sum.item() / top5_acc_count.item()
        losses.avg = losses_sum.item() / losses_count.item()

    # Log
    if (is_master_proc):
        print('\nTest set: Average loss: {:.4f}, Triplet Accuracy: {:.2f}%, Top1 Acc: {:.2f}%, Top5 Acc: {:.2f}%\n'.format(
            losses.avg, 100. * accs.avg, 100. * top1_accs.avg, 100. * top5_accs.avg))

        with open('{}/tnet_checkpoints/val_loss_and_acc.txt'.format(cfg.OUTPUT_PATH), "a") as val_file:
            val_file.write('epoch:{} {:.4f} {:.2f} {:.2f} {:.2f}\n'.format(epoch, losses.avg, 100. * accs.avg, 100. * top1_accs.avg, 100. * top5_accs.avg))

    return accs.avg

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
