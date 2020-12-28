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
from loss.triplet_loss import OnlineTripleLoss
from loss.NCE_loss import NCEAverage, NCEAverage_intra_neg, NCESoftmaxLoss
from clustering.cluster_masks import fit_cluster
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score

#TODO: add this to config file
modality = 'res'
intra_neg = False #True
moco = False #True
neg_type='repeat'


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
        anchor, positive = inputs
        a_target, p_target = targets
        batch_size = torch.tensor(anchor.size(0)).to(device)
        targets = torch.cat((a_target, p_target), 0)

        # Prepare input and send to gpu
        if cfg.MODEL.ARCH == 'slowfast':
            anchor = multipathway_input(anchor, cfg)
            positive = multipathway_input(positive, cfg)
            if cuda:
                for i in range(len(anchor)):
                    anchor[i], positive[i]= anchor[i].to(device), positive[i].to(device)
        elif cuda:
            anchor, positive = anchor.to(device), positive.to(device)

        # Get embeddings of anchors and positives
        anchor_outputs = model(anchor)
        positive_outputs = model(positive)
        outputs = torch.cat((anchor_outputs, positive_outputs), 0)  # dim: [(batch_size * 2), dim_embedding]
        if cuda:
            targets = targets.to(device)

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

    n_parameters = sum([p.data.nelement() for p in model.parameters()])
    if(is_master_proc):
        print('Number of params: {}'.format(n_parameters))

    # Load pretrained backbone if path exists
    if args.pretrain_path is not None:
        model = load_pretrained_model(model, args.pretrain_path, is_master_proc)

    if cfg.MODEL.ARCH == 'uber_nce':
        encoder = model.encoder_q
    else:
        encoder = model

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

    # if cuda:
        encoder = encoder.cuda(device=device)
        if torch.cuda.device_count() > 1:
            #model = nn.DataParallel(model)
            if cfg.MODEL.ARCH == '3dresnet':
                encoder = torch.nn.parallel.DistributedDataParallel(module=encoder,
                    device_ids=[device], find_unused_parameters=True, broadcast_buffers=False)
            else:
                encoder = torch.nn.parallel.DistributedDataParallel(module=encoder,
                    device_ids=[device], broadcast_buffers=False)

    # Load similarity network checkpoint if path exists
    if args.checkpoint_path is not None:
        start_epoch, best_acc = load_checkpoint(model, args.checkpoint_path, is_master_proc)

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
    criterion = OnlineTripleLoss(margin=cfg.LOSS.MARGIN, dist_metric=cfg.LOSS.DIST_METRIC).to(device)
    optimizer = optim.SGD(model.parameters(), lr=cfg.OPTIM.LR, momentum=cfg.OPTIM.MOMENTUM)
    if(is_master_proc):
        print('Using criterion:{} for training'.format(criterion))
        print('Using criterion:{} for validation'.format(val_criterion))

    # ============================== Data Loaders ==============================

    if(is_master_proc):
        print('\n==> Building training data loader (triplet)...')
    train_loader, (train_sampler, _) = data_loader.build_data_loader('train', cfg, is_master_proc, triplets=True)

    if(is_master_proc):
        print('\n==> Building validation data loader (triplet)...')
    val_loader, (_, _) = data_loader.build_data_loader('val', cfg, is_master_proc, triplets=True, negative_sampling=True)

    # Setting is_master_proc to false when loading single video data loaders 
    # deliberately to not re-print data loader information

    if(is_master_proc):
        print('\n==> Building training data loader (single video)...')
    eval_train_loader, (train_data, _) = data_loader.build_data_loader('train',
            cfg, is_master_proc=False, triplets=False, req_train_shuffle=False,
            drop_last=False)

    if(is_master_proc):
        print('\n==> Building validation data loader (single video)...')
    eval_val_loader, (val_data, _) = data_loader.build_data_loader('val', cfg,
            is_master_proc=False, triplets=False, val_sample=None,
            drop_last=False)

    # ============================= Training loop ==============================

    embeddings_computed = False

    for epoch in range(start_epoch, cfg.TRAIN.EPOCHS):
        if (is_master_proc):
            print ('\nEpoch {}/{}'.format(epoch, cfg.TRAIN.EPOCHS-1))

        if args.iterative_cluster and epoch % cfg.ITERCLUSTER.INTERVAL == 0:
            # Get embeddings using current model
            if is_master_proc:
                print('\n=> Computing embeddings')

            if is_master_proc or not embeddings_computed:
                embeddings, true_labels, idxs = get_embeddings_and_labels(args, cfg,
                        encoder, cuda, device, eval_train_loader, split='train',
                        is_master_proc=is_master_proc,
                        load_pkl=embeddings_computed, save_pkl=False)

            if is_master_proc:
                # Cluster
                print('\n=> Clustering')
                start_time = time.time()
                print('embeddings shape', embeddings.size())
                trained_clustering_obj = fit_cluster(embeddings, 'kmeans', cfg.ITERCLUSTER.K)
                print('Time to cluster: {:.2f}s'.format(time.time()-start_time))

                # Calculate NMI for true labels vs cluster assignments
                #true_labels = train_data.get_total_labels()
                NMI = normalized_mutual_info_score(true_labels, trained_clustering_obj.labels_)
                print('NMI between true labels and cluster assignments: {:.3f}\n'.format(NMI))
                with open('{}/tnet_checkpoints/NMIs.txt'.format(cfg.OUTPUT_PATH), "a") as f:
                    f.write('epoch:{} {:.3f}\n'.format(epoch, NMI))

                # Calculate Adjusted NMI for true labels vs cluster assignements
                AMI = adjusted_mutual_info_score(true_labels, trained_clustering_obj.labels_)
                print('AMI between true labels and cluster assignments: {:.3f}\n'.format(AMI))
                with open('{}/tnet_checkpoints/AMIs.txt'.format(cfg.OUTPUT_PATH), "a") as f:
                    f.write('epoch:{} {:.3f}\n'.format(epoch, AMI))

                # Update probability of sampling positive from same video using NMI
                if cfg.ITERCLUSTER.ADAPTIVEP:
                    cfg.DATASET.POSITIVE_SAMPLING_P = float(1.0 - NMI)

                # Get cluster assignments in unshuffled order of dataset
                cluster_assignments_unshuffled_order = [None] * len(eval_train_loader.dataset)
                for i in range(len(trained_clustering_obj.labels_)):
                    cluster_assignments_unshuffled_order[idxs[i]] = trained_clustering_obj.labels_[i]

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
            train_loader, (train_sampler,_) = data_loader.build_data_loader('train', cfg, is_master_proc, triplets=True)

        # Call set_epoch on the distributed sampler so the data is shuffled
        if cfg.NUM_GPUS > 1:
            train_sampler.set_epoch(epoch)

        # Train 
        if cfg.LOSS.TYPE == 'triplet':
            if (is_master_proc): print('==> training with Triplet Loss')
            triplet_train_epoch(train_loader, model, criterion, optimizer, epoch, cfg, cuda, device, is_master_proc)

        elif cfg.LOSS.TYPE == 'contrastive':
            if (is_master_proc): print('==> training with Contrastive Loss')
            n_data = len(train_loader.dataset)
            # n_data = len(cluster_labels) #n_labels
            if intra_neg:
                contrast = NCEAverage_intra_neg(cfg.LOSS.FEAT_DIM, n_data, cfg.LOSS.K, cfg.LOSS.T, cfg.LOSS.M).to(device)
            elif moco:
                contrast = MemoryMoCo(cfg.LOSS.FEAT_DIM, n_data, cfg.LOSS.K, cfg.LOSS.T).to(device)
            else:
                contrast = NCEAverage(cfg.LOSS.FEAT_DIM, n_data, cfg.LOSS.K, cfg.LOSS.T, cfg.LOSS.M).to(device)

            criterion_1 = NCESoftmaxLoss().to(device)
            criterion_2 = NCESoftmaxLoss().to(device)
            if(is_master_proc):
                print('Using criterion:{} for training'.format(criterion_1, criterion_2))
                print('Using criterion:{} for validation'.format(val_criterion))
            contrastive_train_epoch(train_loader, model, criterion_1, criterion_2, contrast, optimizer, epoch, cfg, cuda, device, is_master_proc)
        elif cfg.LOSS.TYPE == 'UberNCE':
            criterion = nn.CrossEntropyLoss().to(device)
            UberNCE_train_epoch(train_loader, model, criterion, optimizer, epoch, cfg, cuda, device, is_master_proc)
        else:
            assert False, 'Loss Type:{} not recognized'.format(cfg.LOSS.TYPE)

        embeddings_computed = False
        # Validate
        if is_master_proc:
            print('\n=> Validating with triplet accuracy and {} top1/5 retrieval on val set with batch_size: {}'.format(cfg.VAL.METRIC, cfg.VAL.BATCH_SIZE))
        acc = validate(val_loader, tripletnet, val_criterion, epoch, cfg, cuda, device, is_master_proc)
        if epoch % 10 == 0:
            if is_master_proc:
                print('\n=> Validating with global top1/5 retrieval from train set with queries from val set')
            topk_acc = k_nearest_embeddings(args, encoder, cuda, device, eval_train_loader, eval_val_loader, train_data, val_data, cfg,
                                        plot=False, epoch=epoch, is_master_proc=is_master_proc)
            embeddings_computed = True
            #if is_master_proc:
            #    print('\n=> Validating with global top1/5 retrieval from train set with queries from train set')
            #top1_acc, _ = k_nearest_embeddings(args, model, cuda, device, eval_train_loader, eval_train_loader, train_data, train_data, cfg, plot=False,
            #                        epoch=epoch, is_master_proc=is_master_proc, out_filename='train_retrieval_acc')

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

    # If iteratively clustering, overwrite the cluster_path
    print('Iteratively clustering?:', args.iterative_cluster)
    if args.iterative_cluster:
        assert(cfg.DATASET.TARGET_TYPE_T == 'cluster_label' and cfg.DATASET.POSITIVE_SAMPLING_P != 1.0)
        cfg.DATASET.CLUSTER_PATH = '{}/vid_clusters.txt'.format(cfg.OUTPUT_PATH)

    print('Multiview positives (25% chance replace): {}'.format(cfg.DATASET.POS_CHANNEL_REPLACE))

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

    # Launch processes for all gpus
    print('\n==> Launching gpu processes...')
    du_helper.launch_processes(args, cfg, func=train, shard_id=shard_id,
        NUM_SHARDS=args.num_shards, ip_address_port=args.ip_address_port)
