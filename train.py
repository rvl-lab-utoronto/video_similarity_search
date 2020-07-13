"""
Created by Sherry Chen on Jul 3, 2020
Build and Train Triplet network. Supports saving and loading checkpoints,
"""

import sys, os

import csv
import argparse
import shutil
import tqdm
import torch
from torch import nn
import torch.optim as optim
from models.triplet_net import Tripletnet
from datasets import data_loader
import torch.backends.cudnn as cudnn

#from pytorch_memlab import MemReporter
from models.model_utils import model_selector, multipathway_input

from config.m_parser import load_config, parse_args

log_interval = 50 #log interval for batch number




cuda = False
if torch.cuda.is_available():
    print('cuda is ready')
    cuda = True
os.environ["CUDA_VISIBLE_DEVICES"]=str('0,1')
device = torch.device('cuda:0')


def load_pretrained_model(model, pretrain_path):
    print('=> loading pretrained model')
    if pretrain_path:
        print('loading pretrained model {}'.format(pretrain_path))
        pretrain = torch.load(pretrain_path, map_location='cpu')
        model.load_state_dict(pretrain['state_dict'])
    print('=> pretrain model:{} is loaded'.format(pretrain_path))
    return model


def save_checkpoint(state, is_best, model_name, output_path, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = "tnet_checkpoints/%s/"%(model_name)
    directory = os.path.join(output_path, directory)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    print('=> checkpoint:{} saved...'.format(filename))
    if is_best:
        shutil.copyfile(filename,  os.path.join(directory, 'model_best.pth.tar'))
        print('=> best_model saved as:{}'.format(os.path.join(directory, 'model_best.pth.tar')))


def load_checkpoint(model, checkpoint_path):
    if os.path.isfile(checkpoint_path):
        print("=> loading checkpoint '{}'".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(checkpoint_path, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(checkpoint_path))

    return start_epoch, best_prec1


def train(train_loader, tripletnet, criterion, optimizer, epoch, cfg):
    losses = AverageMeter()
    accs = AverageMeter()
    emb_norms=AverageMeter()

    triplets = []
    #switching to training mode
    tripletnet.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        anchor, positive, negative = inputs

        (anchor_target, positive_target, negative_target) = targets
        triplets.append([anchor_target, positive_target, negative_target])

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

        #******
        # reporter = MemReporter(tripletnet)
        # print('========== before backward ========')
        # reporter.report()
        #******

        #1 means, dista should be larger than distb
        target = torch.FloatTensor(dista.size()).fill_(-1)
        if cuda:
            target = target.to(device)

        loss_triplet = criterion(dista, distb, target)
        loss_embedd = embedded_x.norm(2) + embedded_y.norm(2) + embedded_z.norm(2)
        loss = loss_triplet + 0.001 *loss_embedd


        #measure accuracy and record loss
        acc = accuracy(dista.cpu(), distb.cpu())
        losses.update(loss_triplet.cpu(), batch_size)
        accs.update(acc, batch_size)
        emb_norms.update(loss_embedd.cpu()/3, batch_size)

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # #******
        # print('========== after backward ========')
        # reporter.report()
        # #******

        torch.cuda.empty_cache()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} | {:.1f}%]\t'
                  'Loss: {:.4f} ({:.4f}) \t'
                  'Acc: {:.2f}% ({:.2f}%) \t'
                  'Emb_Norm: {:.2f} ({:.2f})'.format(
                epoch, batch_idx * batch_size, len(train_loader.dataset), 100. * (batch_idx * batch_size / len(train_loader.dataset)),
                losses.val, losses.avg,
                100. * accs.val, 100. * accs.avg, emb_norms.val, emb_norms.avg))


    with open('{}/tmp_triplets/triplets_{}.txt'.format(cfg.OUTPUT_PATH, epoch), 'w') as f:
        csv_writer = csv.writer(f, delimiter=',')
        csv_writer.writerows(triplets)

    with open('{}/tnet_checkpoints/train_loss_and_acc.txt'.format(cfg.OUTPUT_PATH), "a") as f:
        f.write('{:.4f} {:.2f}\n'.format(losses.avg, 100. * accs.avg))


def validate(val_loader, tripletnet, criterion, epoch, cfg):
    losses = AverageMeter()
    losses_r = AverageMeter()
    accs = AverageMeter()

    tripletnet.eval()
    torch.cuda.empty_cache()
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

            test_loss = criterion(dista, distb, target)
            #add regularization term
            loss_embedd = embedded_x.norm(2) + embedded_y.norm(2) + embedded_z.norm(2)
            loss_r = test_loss + 0.001 *loss_embedd

            # measure accuracy and record loss
            acc = accuracy(dista, distb)
            accs.update(acc.cpu(), batch_size)
            losses.update(test_loss.cpu(), batch_size)
            losses_r.update(loss_r.cpu(), batch_size)

            torch.cuda.empty_cache()

    print('\nTest set: Average loss: {:.4f}({:.4f}), Accuracy: {:.2f}%\n'.format(
        losses.avg, losses_r.avg, 100. * accs.avg))

    with open('{}/tnet_checkpoints/val_loss_and_acc.txt'.format(cfg.OUTPUT_PATH), "a") as val_file:
        val_file.write('{:.4f} {:.4f} {:.2f}\n'.format(losses.avg,losses_r.avg, 100. * accs.avg))

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
    pred = (distb - dista - margin).cpu().data
    return (pred > 0).sum()*1.0/dista.size()[0]


if __name__ == '__main__':
    args = parse_args()
    cfg = load_config(args)

    if not os.path.exists(cfg.OUTPUT_PATH):
        os.makedirs(cfg.OUTPUT_PATH)

    if not os.path.exists(os.path.join(cfg.OUTPUT_PATH, 'tmp_triplets')):
        os.makedirs(os.path.join(cfg.OUTPUT_PATH, 'tmp_triplets'))

    best_acc = 0
    start_epoch = 0
    cudnn.benchmark = True
    # ======================== Similarity Network Setup ========================
    model=model_selector(cfg)
    print('=> finished generating {} backbone model...'.format(cfg.MODEL.ARCH))

    # Load pretrained backbone if path exists
    if args.pretrain_path is not None:
        model = load_pretrained_model(model, pretrain_path)

    tripletnet = Tripletnet(model)

    # Load similarity network checkpoint if path exists
    if args.checkpoint_path is not None:
        start_epoch, best_acc = load_checkpoint(tripletnet, args.checkpoint_path)

    if cuda:
        # if torch.cuda.device_count() > 1:
            # print("Let's use {} GPUs".format(torch.cuda.device_count()))
            # tripletnet = nn.DataParallel(tripletnet, device_ids=[0, 1])
            # print('devices:{}'.format(tripletnet.device_ids))
        tripletnet.to(device)

    print('=> finished generating similarity network...')

    # ============================== Data Loaders ==============================

    train_loader = data_loader.build_data_loader('train', cfg)
    val_loader = data_loader.build_data_loader('val', cfg)

    # ======================== Loss and Optimizer Setup ========================

    criterion = torch.nn.MarginRankingLoss(margin=cfg.LOSS.MARGIN).to(device)
    optimizer = optim.SGD(tripletnet.parameters(), lr=cfg.OPTIM.LR, momentum=cfg.OPTIM.MOMENTUM)

    n_parameters = sum([p.data.nelement() for p in tripletnet.parameters()])
    print(' + Number of params: {}'.format(n_parameters))

    # ============================= Training loop ==============================

    for epoch in range(start_epoch, cfg.TRAIN.EPOCHS):
        train(train_loader, tripletnet, criterion, optimizer, epoch, cfg)
        acc = validate(val_loader, tripletnet, criterion, epoch, cfg)
        print('epoch:{}, acc:{}'.format(epoch, acc))
        is_best = acc > best_acc
        best_acc = max(acc, best_acc)
        save_checkpoint({
            'epoch': epoch+1,
            'state_dict':tripletnet.state_dict(),
            'best_prec1': best_acc,
        }, is_best, cfg.MODEL.ARCH, cfg.OUTPUT_PATH)
