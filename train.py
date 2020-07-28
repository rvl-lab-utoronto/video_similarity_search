"""
Created by Sherry Chen on Jul 3, 2020
Build and Train Triplet network. Supports saving and loading checkpoints,
"""

import sys, os
#import gc
import time
import csv
import argparse
import shutil
import tqdm
import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

#from pytorch_memlab import MemReporter
from models.triplet_net import Tripletnet
from datasets import data_loader
from models.model_utils import model_selector, multipathway_input
from config.m_parser import load_config, arg_parser
import misc.distributed_helper as du_helper

log_interval = 5 #log interval for batch number
offset = 0.00001

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


def train_epoch(train_loader, tripletnet, criterion, optimizer, epoch, cfg, is_master_proc=True):
    triplet_losses = AverageMeter()
    losses_r = AverageMeter()
    accs = AverageMeter()
    emb_norms=AverageMeter()

    # switching to training mode
    tripletnet.train()
    start = time.time()
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

        loss_triplet = criterion(dista, distb, target)
        loss_embedd = embedded_x.norm(2) + embedded_y.norm(2) + embedded_z.norm(2)
        loss = loss_triplet + 0.001 * loss_embedd + offset #adding a small term for numerical stability

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()#create_graph=True)
        optimizer.step()

        # measure accuracy
        acc = accuracy(dista.detach(), distb.detach())

        # Gather the predictions across all the devices
        if (cfg.NUM_GPUS > 1):
            loss_triplet, loss, acc, loss_embedd = du_helper.all_reduce([loss_triplet, loss, acc, loss_embedd])

        # record loss and accuracy
        triplet_losses.update(loss_triplet.detach(), batch_size)
        losses_r.update(loss.detach(), batch_size)
        accs.update(acc, batch_size)
        emb_norms.update(loss_embedd.detach()/3, batch_size)

        if batch_idx % log_interval == 0:
            if (is_master_proc):
                print('Train Epoch: {} [{}/{} | {:.1f}%]\t'
                    'Loss: {:.4f} ({:.4f}) \t'
                    'Acc: {:.2f}% ({:.2f}%) \t'
                    'Emb_Norm: {:.2f} ({:.2f})'.format(
                    epoch, batch_idx * batch_size, len(train_loader.dataset), 100. * (batch_idx * batch_size / len(train_loader.dataset)),
                    triplet_losses.val, triplet_losses.avg,
                    100. * accs.val, 100. * accs.avg, emb_norms.val, emb_norms.avg))

    if (is_master_proc):
        print('epoch:{} runtime:{}'.format(epoch, (time.time()-start)/3600))
        with open('{}/tnet_checkpoints/train_loss_and_acc.txt'.format(cfg.OUTPUT_PATH), "a") as f:
            f.write('epoch:{} runtime:{} {:.4f} {:.4f} {:.2f}\n'.format(epoch, round((time.time()-start)/3600,2), triplet_losses.avg, losses_r.avg, 100. * accs.avg))
            print('saved to file:{}'.format('{}/tnet_checkpoints/train_loss_and_acc.txt'.format(cfg.OUTPUT_PATH)))


def validate(val_loader, tripletnet, criterion, epoch, cfg, is_master_proc=True):
    triplet_losses = AverageMeter()
    losses_r = AverageMeter()
    accs = AverageMeter()

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

            triplet_loss = criterion(dista, distb, target)
            #add regularization term
            loss_embedd = embedded_x.norm(2) + embedded_y.norm(2) + embedded_z.norm(2)
            loss_r = triplet_loss + 0.001 *loss_embedd + offset

            # measure accuracy
            acc = accuracy(dista.detach(), distb.detach())

            # Gather the predictions across all the devices
            if (cfg.NUM_GPUS > 1):
                triplet_loss, loss_r, acc = du_helper.all_reduce([triplet_loss, loss_r, acc])

            # record los and accuracy
            accs.update(acc, batch_size)
            triplet_losses.update(triplet_loss.detach(), batch_size)
            losses_r.update(loss_r.detach(), batch_size)

    if (is_master_proc):
        print('\nTest set: Average loss: {:.4f}({:.4f}), Accuracy: {:.2f}%\n'.format(
            triplet_losses.avg, losses_r.avg, 100. * accs.avg))

        with open('{}/tnet_checkpoints/val_loss_and_acc.txt'.format(cfg.OUTPUT_PATH), "a") as val_file:
            val_file.write('epoch:{} {:.4f} {:.4f} {:.2f}\n'.format(epoch, triplet_losses.avg, losses_r.avg, 100. * accs.avg))

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
    return (pred > 0).sum()*1.0/dista.size()[0]


def train(args, cfg):
    best_acc = 0
    start_epoch = 0
    cudnn.benchmark = True

    global cuda; cuda = torch.cuda.is_available()
    global device; device = torch.cuda.current_device()

    is_master_proc = du_helper.is_master_proc(cfg.NUM_GPUS)

    # ======================== Similarity Network Setup ========================
    model=model_selector(cfg)
    if(is_master_proc):
        print('\n=> finished generating {} backbone model...'.format(cfg.MODEL.ARCH))

    # Load pretrained backbone if path exists
    if args.pretrain_path is not None:
        model = load_pretrained_model(model, args.pretrain_path, is_master_proc)

    tripletnet = Tripletnet(model)
    tripletnet = tripletnet.cuda(device=device)

    if cuda:
        if torch.cuda.device_count() > 1:
            #tripletnet = nn.DataParallel(tripletnet)
            tripletnet = torch.nn.parallel.DistributedDataParallel(module=tripletnet, device_ids=[device])

    # Load similarity network checkpoint if path exists
    if args.checkpoint_path is not None:
        start_epoch, best_acc = load_checkpoint(tripletnet, args.checkpoint_path, is_master_proc)

    if(is_master_proc):
        print('=> finished generating similarity network...')

    # ============================== Data Loaders ==============================

    train_loader, _ = data_loader.build_data_loader('train', cfg, is_master_proc)
    val_loader, _ = data_loader.build_data_loader('val', cfg, is_master_proc)

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
        train_epoch(train_loader, tripletnet, criterion, optimizer, epoch, cfg, is_master_proc)
        acc = validate(val_loader, tripletnet, criterion, epoch, cfg, is_master_proc)
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

    if not os.path.exists(cfg.OUTPUT_PATH):
        os.makedirs(cfg.OUTPUT_PATH)

    if not os.path.exists(os.path.join(cfg.OUTPUT_PATH, 'tmp_triplets')):
        os.makedirs(os.path.join(cfg.OUTPUT_PATH, 'tmp_triplets'))

    if not os.path.exists(os.path.join(cfg.OUTPUT_PATH, 'tnet_checkpoints')):
        os.makedirs(os.path.join(cfg.OUTPUT_PATH, 'tnet_checkpoints'))

    # Set visible gpu devices
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

    if torch.cuda.is_available():
        cfg.NUM_GPUS = torch.cuda.device_count()
        print("Using {} GPU(s) per node".format(torch.cuda.device_count()))

    print ('Node id:', args.shard_id)
    print ('Total nodes:', args.num_shards)

    # Launch processes for all gpus
    du_helper.launch_processes(args, cfg, func=train, shard_id=args.shard_id, NUM_SHARDS=args.num_shards, ip_address_port=args.ip_address_port)
