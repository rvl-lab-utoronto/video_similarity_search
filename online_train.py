"""
Self-supervised training script
"""

def warn(*args, **kwargs):
        pass
import warnings
warnings.warn = warn

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
modality = 'res'
intra_neg = False #True
moco = False #True
neg_type='repeat'

import cv2

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


def UberNCE_train_epoch(train_loader, model, criterion, optimizer, epoch, cfg, cuda, device, is_master_proc=True):
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
            input_seq = input_seq.to(device, non_blocking=True)
            label = labels[0].to(device, non_blocking=True)

        if cfg.MODEL.ARCH == 'info_nce':
            output, target = model(input_seq)
            loss = criterion(output, target)
            top1, top5 = calc_topk_accuracy(output, target, (1,5))

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


def contrastive_train_epoch(train_loader, model, criterion_1, criterion_2, contrast, optimizer, epoch, cfg, cuda, device, is_master_proc=True):
    losses = AverageMeter()
    accs = AverageMeter()
    view1_loss_meter = AverageMeter()
    view2_loss_meter = AverageMeter()
    view1_prob_meter = AverageMeter()
    view2_prob_meter = AverageMeter()

    world_size = du_helper.get_world_size()

    # switching to training mode
    model.train()
    contrast.train()

    # Training loop
    start = time.time()
    for batch_idx, (inputs, labels, index) in enumerate(train_loader):
        view1 = inputs[0]
        if modality=='rgb':
            view2 = inputs[1]
        elif modality == 'res':
            view2 = diff(view1)

        batch_size = torch.tensor(view1.size(0)).to(device)
        # Prepare input and send to gpu
        if cfg.MODEL.ARCH == 'slowfast':
            view1 = multipathway_input(view1, cfg)
            view2 = multipathway_input(view2, cfg)
            if cuda:
                for i in range(len(view1)):
                    view1[i], view2[i]= view1[i].to(device), view2[i].to(device)
        elif cuda:
            view1, view2 = view1.to(device), view2.to(device)

        if cuda:
            labels = labels[0].to(device)
            index = index.to(device)

        # Get embeddings of view1s and view2s
        feat_1 = model(view1)
        feat_2 = model(view2)
        if intra_neg:
            intra_negative = preprocess(view1, neg_type)
            feat_neg = model(intra_negative)
            out_1, out_2 = contrast(feat_1, feat_2, feat_neg, index) #labels
        else:
            out_1, out_2 = contrast(feat_1, feat_2, index) #labels

        view1_loss = criterion_1(out_1)
        view2_loss = criterion_2(out_2)

        view1_prob = out_1[:,0].mean()
        view2_prob = out_2[:,0].mean()

        loss = view1_loss + view2_loss

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
        view1_loss_meter.update(view1_loss.item(), batch_size_world)
        view1_prob_meter.update(view1_prob.item(), batch_size_world)
        view2_loss_meter.update(view1_loss.item(), batch_size_world)
        view2_prob_meter.update(view2_prob.item(), batch_size_world)
        # Log
        if is_master_proc and ((batch_idx + 1) * world_size) % cfg.TRAIN.LOG_INTERVAL == 0:
            print('Train Epoch: {} [{}/{} | {:.1f}%]\t'
                  'Loss: {:.4f} ({:.4f})'.format(epoch, losses.count,
                    len(train_loader.dataset),
                    100. * (losses.count / len(train_loader.dataset)),
                    losses.val, losses.avg))

    if (is_master_proc):
        print('\nTrain set: Average loss: {:.4f}\n'.format(losses.avg))
        print('epoch:{} runtime:{}'.format(epoch, (time.time()-start)/3600))
        with open('{}/tnet_checkpoints/train_loss_and_acc.txt'.format(cfg.OUTPUT_PATH), "a") as f:
            f.write('epoch:{} runtime:{} {:.4f}\n'.format(epoch, round((time.time()-start)/3600,2), losses.avg))
        print('saved to file:{}'.format('{}/tnet_checkpoints/train_loss_and_acc.txt'.format(cfg.OUTPUT_PATH)))


def diff(x):
    shift_x = torch.roll(x, 1, 2)
    return ((x - shift_x) + 1) / 2


def prepare_input(vid_clip, cfg, cuda, device):
    if cfg.MODEL.ARCH == 'slowfast':
        vid_clip = multipathway_input(vid_clip, cfg)
        if cuda:
            for i in range(len(vid_clip)):
                vid_clip[i]= vid_clip[i].to(device)
    elif cuda:
        vid_clip = vid_clip.to(device)

    return vid_clip


def triplet_train_epoch(train_loader, model, criterion, optimizer, epoch, cfg, cuda, device, is_master_proc=True):
    losses = AverageMeter()
    accs = AverageMeter()
    running_n_triplets = AverageMeter()
    world_size = du_helper.get_world_size()
    # switching to training mode
    model.train()

    # Training loop
    start = time.time()
    for batch_idx, (inputs, targets, idx) in enumerate(train_loader):

        if cfg.LOSS.RELATIVE_SPEED_PERCEPTION:
            anchor, positive, fast_positive = inputs
            fast_positive = prepare_input(fast_positive, cfg, cuda, device)
        elif cfg.LOSS.LOCAL_LOCAL_CONTRAST:
            anchor, positive, anchor2 = inputs
            anchor2 = prepare_input(anchor2, cfg, cuda, device)
        elif cfg.LOSS.INTRA_NEGATIVE:
            anchor, positive, intra_neg = inputs 
            intra_neg = prepare_input(intra_neg, cfg, cuda, device)
        else:
            anchor, positive = inputs
        anchor = prepare_input(anchor, cfg, cuda, device)
        positive = prepare_input(positive, cfg, cuda, device)

        a_target, p_target = targets
        batch_size = torch.tensor(anchor.size(0)).to(device)
        targets = torch.cat((a_target, p_target), 0)
        if cuda:
            targets = targets.to(device)

        # Get embeddings of anchors and positives
        if cfg.LOSS.RELATIVE_SPEED_PERCEPTION:
            outputs = model(torch.cat((anchor, positive, fast_positive), 0))
            out_anchor_positive = outputs[:batch_size*2]
            out_anc = outputs[:batch_size]
            out_pos = outputs[batch_size:batch_size*2]
            out_fast_pos = outputs[batch_size*2:batch_size*3]

            # Regular loss
            triplet_loss, n_triplets = criterion(out_anchor_positive, targets, sampling_strategy=cfg.DATASET.SAMPLING_STRATEGY)

            # Relative speed perception loss
            if cfg.LOSS.DIST_METRIC == 'euclidean':
                dist_ap = F.pairwise_distance(out_anc, out_pos, 2)
                dist_an = F.pairwise_distance(out_anc, out_fast_pos, 2)
            elif cfg.LOSS.DIST_METRIC == 'cosine':
                dist_ap = 1 - F.cosine_similarity(out_anc, out_pos, dim=1)
                dist_an = 1 - F.cosine_similarity(out_anc, out_fast_pos, dim=1)

            rsp_criterion = torch.nn.MarginRankingLoss(margin=0.1).to(device)
            target_rsp = torch.FloatTensor(dist_ap.size()).fill_(-1)
            if cuda:
                target_rsp = target_rsp.to(device)
            rsp_loss = rsp_criterion(dist_ap, dist_an, target_rsp)

            # Combined loss
            rsp_lambda = 1.0
            loss = triplet_loss + rsp_loss * rsp_lambda

        elif cfg.LOSS.LOCAL_LOCAL_CONTRAST:
            outputs = model(torch.cat((anchor, positive, anchor2), 0))
            out_anchor_positive = outputs[:batch_size*2]
            out_anc = outputs[:batch_size]
            out_pos = outputs[batch_size:batch_size*2]
            out_anc2 = outputs[batch_size*2:batch_size*3]

            # Regular loss
            triplet_loss, n_triplets = criterion(out_anchor_positive, targets, sampling_strategy=cfg.DATASET.SAMPLING_STRATEGY)

            # Relative speed perception loss
            if cfg.LOSS.DIST_METRIC == 'euclidean':
                dist_ap = F.pairwise_distance(out_anc, out_anc2, 2)
                dist_an = F.pairwise_distance(out_anc, out_pos, 2)
            elif cfg.LOSS.DIST_METRIC == 'cosine':
                dist_ap = 1 - F.cosine_similarity(out_anc, out_anc2, dim=1)
                dist_an = 1 - F.cosine_similarity(out_anc, out_pos, dim=1)

            llc_criterion = torch.nn.MarginRankingLoss(margin=cfg.LOSS.LOCAL_LOCAL_MARGIN).to(device)
            target_llc = torch.FloatTensor(dist_ap.size()).fill_(-1)
            if cuda:
                target_llc = target_llc.to(device)
            llc_loss = llc_criterion(dist_ap, dist_an, target_llc)

            # Combined loss
            llc_lambda = cfg.LOSS.LOCAL_LOCAL_WEIGHT
            loss = triplet_loss + llc_loss * llc_lambda
        
        elif cfg.LOSS.INTRA_NEGATIVE:
            outputs = model(torch.cat((anchor, positive, intra_neg), 0))
            out_anchor_positive = outputs[:batch_size*2]
            out_anc = outputs[:batch_size]
            out_pos = outputs[batch_size:batch_size*2]
            out_intra_neg = outputs[batch_size*2:batch_size*3]

            # Regular loss
            triplet_loss, n_triplets = criterion(out_anchor_positive, targets, sampling_strategy=cfg.DATASET.SAMPLING_STRATEGY)

            # Relative speed perception loss
            if cfg.LOSS.DIST_METRIC == 'euclidean':
                dist_ap = F.pairwise_distance(out_anc, out_intra_neg, 2)
                dist_an = F.pairwise_distance(out_anc, out_pos, 2)
            elif cfg.LOSS.DIST_METRIC == 'cosine':
                dist_ap = 1 - F.cosine_similarity(out_anc, out_intra_neg, dim=1)
                dist_an = 1 - F.cosine_similarity(out_anc, out_pos, dim=1)

            intra_neg_criterion = torch.nn.MarginRankingLoss(margin=0.04).to(device)
            target_intra_neg = torch.FloatTensor(dist_ap.size()).fill_(-1)
            if cuda:
                target_llc = target_intra_neg.to(device)
            intra_neg_loss = intra_neg_criterion(dist_ap, dist_an, target_llc)

            # Combined loss
            intra_neg_lambda = 0.4
            loss = triplet_loss + intra_neg_loss * intra_neg_lambda


            # h, w, l = anchor[0][0].shape
            # size = (w, h)
            # anchor_out = cv2.VideoWriter('pos.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
            # for i in range(len(anchor[0])):
            #     img = np.uint8(anchor[0][i].cpu().detach())
            #     anchor_out.write(img)
            # anchor_out.release()

            
        else:
            outputs = model(torch.cat((anchor, positive), 0))  # dim: [(batch_size * 2), dim_embedding]
            # Sample negatives from batch for each anchor/positive and compute loss
            loss, n_triplets = criterion(outputs, targets, sampling_strategy=cfg.DATASET.SAMPLING_STRATEGY)

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
        running_n_triplets.update(n_triplets)

        # Log
        if is_master_proc and ((batch_idx + 1) * world_size) % cfg.TRAIN.LOG_INTERVAL == 0:
            if cfg.LOSS.RELATIVE_SPEED_PERCEPTION:
                print('Train Epoch: {} [{}/{} | {:.1f}%]\t'
                      'Loss: {:.4f} ({:.4f}) \t'
                      'Triplet loss: {:.4f} \t'
                      'RSP loss: {:.4f} \t'
                      'N_Triplets: {:.1f}'.format(epoch, losses.count,
                        len(train_loader.dataset),
                        100. * (losses.count / len(train_loader.dataset)),
                        losses.val, losses.avg,
                        triplet_loss, rsp_loss,
                        running_n_triplets.avg))

            elif cfg.LOSS.LOCAL_LOCAL_CONTRAST:
                print('Train Epoch: {} [{}/{} | {:.1f}%]\t'
                      'Loss: {:.4f} ({:.4f}) \t'
                      'Triplet loss: {:.4f} \t'
                      'llc loss: {:.4f} \t'
                      'N_Triplets: {:.1f}'.format(epoch, losses.count,
                        len(train_loader.dataset),
                        100. * (losses.count / len(train_loader.dataset)),
                        losses.val, losses.avg,
                        triplet_loss, llc_loss,
                        running_n_triplets.avg))

            else:
                print('Train Epoch: {} [{}/{} | {:.1f}%]\t'
                      'Loss: {:.4f} ({:.4f}) \t'
                      'N_Triplets: {:.1f}'.format(epoch, losses.count,
                        len(train_loader.dataset),
                        100. * (losses.count / len(train_loader.dataset)),
                        losses.val, losses.avg, running_n_triplets.avg))

    if (is_master_proc):
        print('\nTrain set: Average loss: {:.4f}\n'.format(losses.avg))
        print('epoch:{} runtime:{}'.format(epoch, (time.time()-start)/3600))
        with open('{}/tnet_checkpoints/train_loss_and_acc.txt'.format(cfg.OUTPUT_PATH), "a") as f:
            f.write('epoch:{} runtime:{} {:.4f}\n'.format(epoch, round((time.time()-start)/3600,2), losses.avg))
        print('saved to file:{}'.format('{}/tnet_checkpoints/train_loss_and_acc.txt'.format(cfg.OUTPUT_PATH)))


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
        print('\n==> Generating {} backbone model (for training)...'.format(cfg.MODEL.ARCH))


    model=model_selector(cfg, is_master_proc=is_master_proc)

    ## SyncBatchNorm
    if cfg.SYNC_BATCH_NORM:
        print('Converting BatchNorm*D to SyncBatchNorm!')
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    n_parameters = sum([p.data.nelement() for p in model.parameters()])
    if(is_master_proc):
        print('Number of params: {}'.format(n_parameters))

    if cfg.MODEL.ARCH == 'uber_nce':
        encoder = model.encoder_q
    else:
        encoder = model

    def DDP(model):
        # Transfer model to DDP
        model = model.cuda(device=device)
        if torch.cuda.device_count() > 1:
            #model = nn.DataParallel(model)
            if cfg.MODEL.ARCH == '3dresnet':
                model = torch.nn.parallel.DistributedDataParallel(module=model,
                    device_ids=[device],
                    #find_unused_parameters=True,
                    #broadcast_buffers=False)
                    )
            else:
                model = torch.nn.parallel.DistributedDataParallel(module=model,
                    device_ids=[device],
                    #broadcast_buffers=False)
                    )
        return model

    # Load similarity network checkpoint if path exists

    if args.vector:
        load_path = "tnet_checkpoints/%s/checkpoint.pth.tar"%(cfg.MODEL.ARCH)
        load_path = os.path.join(args.checkpoint_path, load_path)
    else:
        load_path = args.checkpoint_path

    if args.checkpoint_path is not None and os.path.exists(load_path):
        start_epoch, best_acc = load_checkpoint(model, load_path, is_master_proc)

    if cuda:
        model = DDP(model)
        encoder = DDP(encoder)

    # Load pretrained backbone if path exists
    if args.pretrain_path is not None:
        model = load_pretrained_model(model, args.pretrain_path, is_master_proc)

    # Setup tripletnet used for validation
    if(is_master_proc):
        print('\n==> Generating triplet network (for validation) ...')

    #only for validation
    tripletnet = Tripletnet(encoder, cfg.LOSS.DIST_METRIC)

    if cuda:
        tripletnet = tripletnet.cuda(device=device)
        if torch.cuda.device_count() > 1:
            if cfg.MODEL.ARCH == '3dresnet':
                tripletnet = torch.nn.parallel.DistributedDataParallel(module=tripletnet,
                    device_ids=[device], find_unused_parameters=True)
            else:
                tripletnet = torch.nn.parallel.DistributedDataParallel(module=tripletnet, device_ids=[device])

    # ======================== Loss and Optimizer Setup ========================
    if(is_master_proc):
        print('\n==> Setting criterion...')
    val_criterion = torch.nn.MarginRankingLoss(margin=cfg.LOSS.MARGIN).to(device)

    criterion = OnlineTripletLoss(margin=cfg.LOSS.MARGIN, dist_metric=cfg.LOSS.DIST_METRIC).to(device)
    # criterion = MemTripletLoss(margin=cfg.LOSS.MARGIN, dist_metric=cfg.LOSS.DIST_METRIC).to(device) #MemTripletLoss

    if cfg.OPTIM.OPTIMIZER == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=cfg.OPTIM.LR, weight_decay=cfg.OPTIM.WD)
    elif cfg.OPTIM.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=cfg.OPTIM.LR, momentum=cfg.OPTIM.MOMENTUM)
    else:
        print('{} optimizer not supported'.format(cfg.OPTIM.OPTIMIZER))

    if(is_master_proc):
        print('Using {} optimizer with lr={}'.format(cfg.OPTIM.OPTIMIZER, cfg.OPTIM.LR))
        if cfg.OPTIM.OPTIMIZER == 'adam':
            print('Weight decay = {}'.format(cfg.OPTIM.WD))
        elif cfg.OPTIM.OPTIMIZER == 'sgd':
            print('Momentum = {}'.format(cfg.OPTIM.MOMENTUM))
        print('Using criterion:{} for training'.format(criterion))
        print('Using criterion:{} for validation'.format(val_criterion))

    # ============================== Data Loaders ==============================
    
    if args.start_epoch != None:
        start_epoch = args.start_epoch

    m_iter_cluster = False
    if args.iterative_cluster:
        if start_epoch >= cfg.ITERCLUSTER.WARMUP_EPOCHS:
            m_iter_cluster = True
            cfg.DATASET.CLUSTER_PATH = '{}/vid_clusters.txt'.format(cfg.OUTPUT_PATH)

    if not m_iter_cluster or start_epoch != 0:
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
    eval_train_loader, (train_data, _) = data_loader.build_data_loader('train',
            cfg, is_master_proc, triplets=False, req_train_shuffle=False,
            drop_last=False)

    if(is_master_proc):
        print('\n==> Building validation data loader (single video)...')
    eval_val_loader, (val_data, _) = data_loader.build_data_loader('val', cfg,
            is_master_proc, triplets=False, val_sample=None,
            drop_last=False, batch_size=1)

    # ============================= Training loop ==============================

    embeddings_computed = False

    for epoch in range(start_epoch, cfg.TRAIN.EPOCHS):
        if (is_master_proc):
            print ('\nEpoch {}/{}'.format(epoch, cfg.TRAIN.EPOCHS-1))

        # =================== Compute embeddings and cluster ===================

        if args.iterative_cluster and epoch == cfg.ITERCLUSTER.WARMUP_EPOCHS:
            m_iter_cluster = True
            cfg.DATASET.CLUSTER_PATH = '{}/vid_clusters.txt'.format(cfg.OUTPUT_PATH)

        if m_iter_cluster and epoch % cfg.ITERCLUSTER.INTERVAL == 0:
            # Get embeddings using current model
            if is_master_proc:
                print('\n=> Computing embeddings')

            if is_master_proc or not embeddings_computed:
                start_time = time.time()
                embeddings, true_labels, idxs = get_embeddings_and_labels(args, cfg,
                        encoder, cuda, device, eval_train_loader, split='train',
                        is_master_proc=is_master_proc,
                        load_pkl=embeddings_computed, save_pkl=False)
                if is_master_proc:
                    print('Time to get embeddings: {:.2f}s'.format(time.time()-start_time))

            if is_master_proc:
                # Cluster
                print('\n=> Clustering')
                start_time = time.time()
                print('embeddings shape', embeddings.size())

                cluster_labels = fit_cluster(embeddings, cfg.ITERCLUSTER.METHOD,
                                        cfg.ITERCLUSTER.K, cfg.ITERCLUSTER.L2_NORMALIZE,
                                        cfg.ITERCLUSTER.FINCH_PARTITION)

                print('Time to cluster: {:.2f}s'.format(time.time()-start_time))

                # Calculate NMI for true labels vs cluster assignments
                #true_labels = train_data.get_total_labels()
                NMI = normalized_mutual_info_score(true_labels, cluster_labels)
                print('NMI between true labels and cluster assignments: {:.3f}'.format(NMI))
                with open('{}/tnet_checkpoints/NMIs.txt'.format(cfg.OUTPUT_PATH), "a") as f:
                    f.write('epoch:{} {:.3f}\n'.format(epoch, NMI))

                # Calculate Adjusted NMI for true labels vs cluster assignements
                AMI = adjusted_mutual_info_score(true_labels, cluster_labels)
                print('AMI between true labels and cluster assignments: {:.3f}\n'.format(AMI))
                with open('{}/tnet_checkpoints/AMIs.txt'.format(cfg.OUTPUT_PATH), "a") as f:
                    f.write('epoch:{} {:.3f}\n'.format(epoch, AMI))

                # Update probability of sampling positive from same video using NMI
                if cfg.ITERCLUSTER.ADAPTIVEP:
                    cfg.DATASET.POSITIVE_SAMPLING_P = float(1.0 - NMI)

                # Get cluster assignments in unshuffled order of dataset
                cluster_assignments_unshuffled_order = [None] * len(eval_train_loader.dataset)
                for i in range(len(cluster_labels)):
                    cluster_assignments_unshuffled_order[idxs[i]] = cluster_labels[i]

                # Save cluster assignments corresponding to unshuffled order of dataset
                cluster_output_path = os.path.join(cfg.OUTPUT_PATH, 'vid_clusters.txt')
                with open(cluster_output_path, "w") as f:
                    for label in cluster_assignments_unshuffled_order:
                        f.write('{}\n'.format(label))
                print('Saved cluster labels to', cluster_output_path)

            # Make processes wait for master process to finish with clustering
            if cfg.NUM_GPUS > 1:
                torch.distributed.barrier()

            # Rebuild train_loader with new cluster assignments as pseudolabels
            if(is_master_proc):
                print('\n==> Building training data loader (triplet)...')
            train_loader, (_, train_sampler) = data_loader.build_data_loader('train', cfg, is_master_proc, triplets=True)

        # ====================== Training for this epoch =======================

        # Call set_epoch on the distributed sampler so the data is shuffled
        if cfg.NUM_GPUS > 1:
            train_sampler.set_epoch(epoch)

        # Train
        if cfg.LOSS.TYPE == 'triplet':
            if (is_master_proc):
                print('==> training with Triplet Loss with criterion:{}'.format(criterion))
            if cfg.DATASET.MODALITY:
                assert False, 'not supported'
                #triplet_multiview_train_epoch(train_loader, model, criterion, optimizer, epoch, cfg, cuda, device, is_master_proc)
            elif cfg.MODEL.PREDICT_TEMPORAL_DS:
                assert False, 'not supported'
                #triplet_temporal_train_epoch(train_loader, model, criterion, optimizer, epoch, cfg, cuda, device, is_master_proc)

            else:
                triplet_train_epoch(train_loader, model, criterion, optimizer, epoch,
                                    cfg, cuda, device, is_master_proc)

        elif cfg.LOSS.TYPE == 'contrastive':
            if (is_master_proc): print('\n==> Training with Contrastive Loss')
            n_data = len(train_loader.dataset)
            # n_data = len(cluster_labels) #n_labels
            if intra_neg:
                contrast = NCEAverage_intra_neg(cfg.LOSS.FEAT_DIM, n_data,
                        cfg.LOSS.K, cfg.LOSS.T, cfg.LOSS.M).to(device)
            elif moco:
                contrast = MemoryMoCo(cfg.LOSS.FEAT_DIM, n_data, cfg.LOSS.K,
                        cfg.LOSS.T).to(device)
            else:
                contrast = NCEAverage(cfg.LOSS.FEAT_DIM, n_data, cfg.LOSS.K,
                        cfg.LOSS.T, cfg.LOSS.M).to(device)

            criterion_1 = NCESoftmaxLoss().to(device)
            criterion_2 = NCESoftmaxLoss().to(device)
            if(is_master_proc):
                print('Using criterion:{} for training'.format(criterion_1, criterion_2))
                print('Using criterion:{} for validation'.format(val_criterion))
            contrastive_train_epoch(train_loader, model, criterion_1, criterion_2,
                    contrast, optimizer, epoch, cfg, cuda, device, is_master_proc)

        elif cfg.LOSS.TYPE == 'UberNCE':
            criterion = nn.CrossEntropyLoss().to(device)
            UberNCE_train_epoch(train_loader, model, criterion, optimizer,
                    epoch, cfg, cuda, device, is_master_proc)

        else:
            assert False, 'Loss Type:{} not recognized'.format(cfg.LOSS.TYPE)

        # Old embeddings are now obsolete
        embeddings_computed = False

        # ============================= Evaluation =============================

        # Validate with triplet loss and retrieval on val set with query from val set
        if is_master_proc:
            print('\n=> Validating with triplet accuracy and {} top1/5 retrieval on val set with batch_size: {}'.format(cfg.VAL.METRIC, cfg.VAL.BATCH_SIZE))
            print('=> Using criterion:{} for validation'.format(val_criterion))

        acc = validate(val_loader, tripletnet, val_criterion, epoch, cfg, cuda,
                       device, is_master_proc)

        is_best = False

        if epoch % 10 == 0:
            if is_master_proc:
                print('\n=> Validating with global top1/5 retrieval from train set with queries from val set')
            topk_acc = k_nearest_embeddings(args, encoder, cuda, device, eval_train_loader,
                                            eval_val_loader, train_data, val_data, cfg,
                                            plot=False, epoch=epoch, is_master_proc=is_master_proc)
            embeddings_computed = True

            # Update best accuracy
            if is_master_proc:
                top1_acc = topk_acc[0]
                is_best = top1_acc > best_acc
                best_acc = max(top1_acc, best_acc)

        # Save checkpoint

        if torch.cuda.device_count() > 1:
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()

        if not args.vector or (args.vector and (epoch % 100 == 0 or is_best or epoch == cfg.TRAIN.EPOCHS - 1)):
            save_checkpoint({
                'epoch': epoch+1,
                'state_dict': state_dict,
                'best_prec1': best_acc,
            }, is_best, cfg.MODEL.ARCH, cfg.OUTPUT_PATH, is_master_proc)

            if epoch % 200 == 0:
                filename = 'checkpoint_{}.pth.tar'.format(epoch)
                save_checkpoint({
                    'epoch': epoch+1,
                    'state_dict': state_dict,
                    'best_prec1': best_acc,
                }, is_best, cfg.MODEL.ARCH, cfg.OUTPUT_PATH, is_master_proc, filename)

        if args.vector:
            save_checkpoint({
                'epoch': epoch+1,
                'state_dict': state_dict,
                'best_prec1': best_acc,
            }, is_best, cfg.MODEL.ARCH, args.checkpoint_path, is_master_proc)


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
