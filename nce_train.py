import sys, os
#import gc
import numpy as np
import time
import argparse
import tqdm
import torch
from torch import nn
import torch.nn.functional as F 
import torch.optim as optim
import torch.backends.cudnn as cudnn
from validation import validate
from evaluate import k_nearest_embeddings
from models.triplet_net import Tripletnet
from datasets import data_loader
from datasets.gen_neg import preprocess
from models.model_utils import (model_selector, multipathway_input,
                            load_pretrained_model, save_checkpoint, load_checkpoint,
                            AverageMeter, accuracy, create_output_dirs)
from config.m_parser import load_config, arg_parser
import misc.distributed_helper as du_helper
from loss.triplet_loss import OnlineTripleLoss
from loss.NCE_loss import NCEAverage, NCEAverage_intra_neg, NCESoftmaxLoss

def calc_topk_accuracy(output, target, topk=(1,)):
    """
    Modified from: https://gist.github.com/agermanidis/275b23ad7a10ee89adccf021536bb97e
    Given predicted and ground truth labels, 
    calculate top-k accuracies.
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1 / batch_size))
    return res

def calc_mask_accuracy(output, target_mask, topk=(1,)):
    maxk = max(topk)
    _, pred = output.topk(maxk,1,True,True)

    zeros = torch.zeros_like(target_mask).long()
    pred_mask = torch.zeros_like(target_mask).long()

    res = []
    for k in range(maxk):
        pred_ = pred[:,k].unsqueeze(1)
        onehot = zeros.scatter(1,pred_,1)
        pred_mask = onehot + pred_mask # accumulate 
        if k+1 in topk:
            res.append(((pred_mask * target_mask).sum(1)>=1).float().mean(0))
    return res 


def train_epoch(train_loader, model, criterion, optimizer, epoch, cfg, cuda, device, is_master_proc=True):
    losses = AverageMeter()
    accs = AverageMeter()
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()

    world_size = du_helper.get_world_size()
    
    # switching to training mode
    model.train()

    def tr(x):
        # print(x.shape)
        B = x.shape[0]
        x = torch.tensor(x)
        return x.view(B, 3, 2, cfg.DATA.SAMPLE_DURATION, cfg.LOSS.FEAT_DIM, cfg.LOSS.FEAT_DIM).transpose(1,2).contiguous() #TODO: make it configureable
    
    # Training loop
    start = time.time()
    for batch_idx, (inputs, labels, index) in enumerate(train_loader):
        inputs = np.concatenate(inputs, axis=1) # [ B, N, C, W, H]
        input_seq = tr(inputs)        
        batch_size = torch.tensor(input_seq.size(0)).to(device)
        
        if cuda:
            input_seq = input_seq.to(device)
            label = labels[0].to(device)

        if cfg.MODEL.ARCH == 'info_nce':
            output, target = model(input_seq)
            loss = criterion(output, target)
            top1, top5 = calc_topk_accuracy(output, target, (1,5))

            # print('loss', loss)
        if cfg.MODEL.ARCH == 'uber_nce':
            # optimize all positive pairs, compute the mean for num_pos and for batch_size 
            output, target = model(input_seq, label)
            loss = - (F.log_softmax(output, dim=1) * target).sum(1) / target.sum(1)
            loss = loss.mean()
            top1, top5 = calc_mask_accuracy(output, target, (1,5))

        # Compute gradient and perform optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Average loss across all gpu processes
        if cfg.NUM_GPUS > 1:
            [loss] = du_helper.all_reduce([loss], avg=True)
            [batch_size_world] = du_helper.all_reduce([batch_size], avg=False)
        else:
            batch_size_world = batch_size

        batch_size_world = batch_size_world.item()

        # Update running loss
        losses.update(loss.item(), batch_size_world)
        top1_meter.update(top1.item(), batch_size)
        top5_meter.update(top5.item(), batch_size)

        # Log
        if is_master_proc and ((batch_idx + 1) * world_size) % cfg.TRAIN.LOG_INTERVAL == 0:
            print('Train Epoch: {} [{}/{} | {:.1f}%]\t'
                  'Loss: {:.4f} ({:.4f})  Top1:{} Top5:{}'.format(epoch, losses.count,
                    len(train_loader.dataset),
                    100. * (losses.count / len(train_loader.dataset)),
                    losses.val, losses.avg, top1_meter.val, top5_meter.val))

    if (is_master_proc):
        print('\nTrain set: Average loss: {:.4f}\n'.format(losses.avg))
        print('epoch:{} runtime:{}'.format(epoch, (time.time()-start)/3600))
        with open('{}/tnet_checkpoints/train_loss_and_acc.txt'.format(cfg.OUTPUT_PATH), "a") as f:
            f.write('epoch:{} runtime:{} {:.4f}\n'.format(epoch, round((time.time()-start)/3600,2), losses.avg))
        print('saved to file:{}'.format('{}/tnet_checkpoints/train_loss_and_acc.txt'.format(cfg.OUTPUT_PATH)))
    return top1_meter.avg, top5_meter.avg


def diff(x):
    shift_x = torch.roll(x, 1, 2)
    return ((x - shift_x) + 1) / 2

def adjust_learning_rate(optimizer, epoch, cfg):
    """Decay the learning rate based on schedule"""
    lr = cfg.OPTIM.LR
    # stepwise lr schedule
    for milestone in cfg.OPTIM.SCHEDULE:
        lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# =========================== Running Training Loop ========================== #

# Setup training and run training loop
def train(args, cfg):
    best_acc = 0
    start_epoch = 0
    cudnn.benchmark = True

    # Check if this is the master process (true if not distributed)
    is_master_proc = du_helper.is_master_proc(cfg.NUM_GPUS)

    # Cuda and current device
    cuda = torch.cuda.is_available()
    device = torch.cuda.current_device()

    if is_master_proc:
        create_output_dirs(cfg)

    # ======================== Similarity Network Setup ========================

    # Select appropriate model
    if(is_master_proc):
        print('\n==> Generating {} backbone model...'.format(cfg.MODEL.ARCH))
    model=model_selector(cfg)
    n_parameters = sum([p.data.nelement() for p in model.parameters()])
    if(is_master_proc):
        print('Number of params: {}'.format(n_parameters))

    # Load pretrained backbone if path exists
    if args.pretrain_path is not None:
        model = load_pretrained_model(model, args.pretrain_path, is_master_proc)

    # Transfer model to DDP
    if cuda:
        model = model.cuda(device=device)
        if torch.cuda.device_count() > 1:
            #model = nn.DataParallel(model)
            if cfg.MODEL.ARCH == '3dresnet':
                model = torch.nn.parallel.DistributedDataParallel(module=model,
                    device_ids=[device], find_unused_parameters=True, broadcast_buffers=False)
            else:
                model = torch.nn.parallel.DistributedDataParallel(module=model,
                    device_ids=[device], broadcast_buffers=False)

    # Load similarity network checkpoint if path exists
    if args.checkpoint_path is not None:
        start_epoch, best_acc = load_checkpoint(model, args.checkpoint_path, is_master_proc)
    
   
    # Setup tripletnet used for validation
    if(is_master_proc):
        print('\n==> Generating triplet network...')
    tripletnet = Tripletnet(model, cfg.LOSS.DIST_METRIC)
    if cuda:
        tripletnet = tripletnet.cuda(device=device)
        if torch.cuda.device_count() > 1:
            if cfg.MODEL.ARCH == '3dresnet':
                tripletnet = torch.nn.parallel.DistributedDataParallel(module=tripletnet,
                    device_ids=[device], find_unused_parameters=True)
            else:
                tripletnet = torch.nn.parallel.DistributedDataParallel(module=tripletnet, device_ids=[device])



    # ============================== Data Loaders ==============================

    if(is_master_proc):
        print('\n==> Building training data loader (triplet)...')
    train_loader, (_, train_sampler) = data_loader.build_data_loader('train', cfg, is_master_proc, triplets=True)

    if(is_master_proc):
        print('\n==> Building validation data loader (triplet)...')
    val_loader, (_, _) = data_loader.build_data_loader('val', cfg, is_master_proc, triplets=True, negative_sampling=True)

    # Setting is_master_proc to false when loading single video data loaders 
    # deliberately to not re-print data loader information

    if(is_master_proc):
        print('\n==> Building training data loader (single video)...')
    eval_train_loader, (train_data, _) = data_loader.build_data_loader('train', cfg, is_master_proc=False, triplets=False)

    if(is_master_proc):
        print('\n==> Building validation data loader (single video)...')
    eval_val_loader, (val_data, _) = data_loader.build_data_loader('val', cfg, is_master_proc=False, triplets=False, val_sample=None)


    # ======================== Loss and Optimizer Setup ========================

    if(is_master_proc):
        print('\n==> Setting criterion & contrastive...')
    val_criterion = torch.nn.MarginRankingLoss(margin=cfg.LOSS.MARGIN).to(device)
    n_data = len(train_loader.dataset)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=cfg.OPTIM.LR, momentum=cfg.OPTIM.MOMENTUM)
    if(is_master_proc):
        print('Using criterion:{} for training'.format(criterion))
        # print('Using criterion:{} for validation'.format(val_criterion))
    

    # ============================= Training loop ==============================

    for epoch in range(start_epoch, cfg.TRAIN.EPOCHS):
        if (is_master_proc):
            print ('\nEpoch {}/{}'.format(epoch, cfg.TRAIN.EPOCHS-1))

        # Call set_epoch on the distributed sampler so the data is shuffled
        if cfg.NUM_GPUS > 1:
            train_sampler.set_epoch(epoch)

        adjust_learning_rate(optimizer, epoch, cfg)
        top1_acc, top5_acc = train_epoch(train_loader, model, criterion, optimizer, epoch, cfg, cuda, device, is_master_proc)
        acc =  top1_acc
        # Validate
        # if is_master_proc:
        #     print('\n=> Validating with triplet accuracy and {} top1/5 retrieval on val set with batch_size: {}'.format(cfg.VAL.METRIC, cfg.VAL.BATCH_SIZE))
        # acc = validate(val_loader, tripletnet, val_criterion, epoch, cfg, cuda, device, is_master_proc)
        # if epoch+1 % 10 == 0:
        #     if is_master_proc:
        #         print('\n=> Validating with global top1/5 retrieval from train set with queries from val set')
        #     topk_acc = k_nearest_embeddings(args, model, cuda, device, eval_train_loader, eval_val_loader, train_data, val_data, cfg, 
        #                                 plot=False, epoch=epoch, is_master_proc=is_master_proc)
            
        # Update best accuracy and save checkpoint
        is_best = acc > best_acc
        best_acc = max(acc, best_acc)
        save_checkpoint({
            'epoch': epoch+1,
            'state_dict':model.state_dict(),
            'best_prec1': best_acc,
        }, is_best, cfg.MODEL.ARCH, cfg.OUTPUT_PATH, is_master_proc)


if __name__ == '__main__':
    print ('\n==> Parsing parameters:')
    args = arg_parser().parse_args()
    cfg = load_config(args)

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
    print('Probability of sampling view2 from same video: {}'.format(cfg.DATASET.POSITIVE_SAMPLING_P))
    print('OUTPUT_PATH is set to: {}'.format(cfg.OUTPUT_PATH))
    print('BATCH_SIZE is set to: {}'.format(cfg.TRAIN.BATCH_SIZE))
    print('NUM_WORKERS is set to: {}'.format(cfg.TRAIN.NUM_DATA_WORKERS))
    print('SAMPLE SIZE is set to: {}'.format(cfg.DATA.SAMPLE_SIZE))
    print('N_CLASSES is set to: {}'.format(cfg.RESNET.N_CLASSES))
    print('Learning rate is set to {}'.format(cfg.OPTIM.LR))

    # Launch processes for all gpus
    print('\n==> Launching gpu processes...')
    du_helper.launch_processes(args, cfg, func=train, shard_id=shard_id,
        NUM_SHARDS=args.num_shards, ip_address_port=args.ip_address_port)
