"""Video retrieval experiment, top-k."""
import os
import math
import itertools
import argparse
import time
import random
import json

from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics as metrics
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances

from iic_datasets.ucf101 import UCF101ClipRetrievalDataset
from iic_datasets.hmdb51 import HMDB51ClipRetrievalDataset
from iic_datasets.CoCLR_model import LinearClassifier
# from models.c3d import C3D
# from models.r3d import R3DNet
# from models.r21d import R2Plus1DNet
from models.triplet_net import Tripletnet
from models.model_utils import model_selector, multipathway_input, load_checkpoint, load_pretrained_model
from config.m_parser import load_config, arg_parser
from config.default_params import get_cfg

def load_pretrained_weights(ckpt_path):
    """load pretrained weights and adjust params name."""
    adjusted_weights = {}
    pretrained_weights = torch.load(ckpt_path)
    for name, params in pretrained_weights.items():
        if 'base_network' in name:
            name = name[name.find('.')+1:]
            adjusted_weights[name] = params
            print('Pretrained weight name: [{}]'.format(name))
    return adjusted_weights

class Normalize(transforms.Normalize):

    def randomize_parameters(self):
        pass


# Return normalization function used per image in a video
def get_normalize_method(mean, std, no_mean_norm, no_std_norm, num_channels=3, is_master_proc=True):
    if no_mean_norm:
        mean = [0, 0, 0]
    elif no_std_norm:
        std = [1, 1, 1]

    extra_num_channel = num_channels-3
    mean.extend([0] * extra_num_channel)
    std.extend([1] * extra_num_channel)

    print(extra_num_channel, mean, std)
    if (is_master_proc):
        print('Normalize mean:{}, std:{}'.format(mean, std))
    return Normalize(mean, std)


# Return mean and std deviation used for normalization
def get_mean_std(value_scale, dataset):
    if dataset == 'kinetics':
        mean = [0.4345, 0.4051, 0.3775]
        std = [0.2768, 0.2713, 0.2737]
    else:
        #mean = [0.5, 0.5, 0.5]
        #std = [0.5, 0.5, 0.5]
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]

    mean = [x * value_scale for x in mean]
    std = [x * value_scale for x in std]

    return mean, std




def neq_load_customized(model, pretrained_dict, verbose=True):
    ''' load pre-trained model in a not-equal way,
    when new model has been partially modified '''
    model_dict = model.state_dict()
    tmp = {}
    if verbose:
        print('\n=======Check Weights Loading======')
        print('Weights not used from pretrained file:')
        for k, v in pretrained_dict.items():
            if k in model_dict:
                tmp[k] = v
            else:
                print(k)
        print('---------------------------')
        print('Weights not loaded into new model:')
        for k, v in model_dict.items():
            if k not in pretrained_dict:
                print(k)
        print('===================================\n')
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    del pretrained_dict
    model_dict.update(tmp)
    del tmp
    model.load_state_dict(model_dict)
    return model

def extract_feature(args, split='train'):
    """Extract and save features for train split, several clips per video."""
    torch.backends.cudnn.benchmark = True
    # Force the pytorch to create context on the specific device 
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    ########### model ##############
    if args.model == 'CoCLR':
         ### classifier model ###
        num_class_dict = {'ucf101':   101, 'hmdb51':   51}
        num_class = num_class_dict[args.dataset]

        model = LinearClassifier(
                    network='s3d', 
                    num_class=num_class,
                    dropout=0.9,
                    use_dropout=False,
                    use_final_bn=True,
                    use_l2_norm=True)

        params = []
        for name, param in model.named_parameters():
            if 'backbone' in name:
                param.requires_grad = False
            else: 
                params.append({'params': param})
        
        print('\n===========Check Grad============')
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name, param.requires_grad)
        print('=================================\n')

        # model = torch.nn.DataParallel(model)
        # model = model.module

        if os.path.isfile(args.checkpoint_path):
            print("=> loading testing checkpoint '{}'".format(args.checkpoint_path))
            checkpoint = torch.load(args.checkpoint_path, map_location=torch.device('cpu'))
            epoch = checkpoint['epoch']
            state_dict = checkpoint['state_dict']

            # if args.retrieval: # if directly test on pretrained network
            new_dict = {}
            for k,v in state_dict.items():
                k = k.replace('encoder_q.0.', 'backbone.')
                new_dict[k] = v
            state_dict = new_dict
            # model.load_state_dict(state_dict)
            neq_load_customized(model, state_dict, verbose=False)
            # try: model_without_dp.load_state_dict(state_dict)
            # except: neq_load_customized(model_without_dp, state_dict, verbose=True)



        
    else:
        cfg = get_cfg()
        cfg.merge_from_file(args.cfg_file)

        model=model_selector(cfg, projection_head=True)

        # Load similarity network checkpoint if path exists
        if args.checkpoint_path is not None:
            start_epoch, best_acc = load_checkpoint(model, args.checkpoint_path)

    global cuda; cuda = torch.cuda.is_available()

    if cuda:
        #model = DDP(model)
        if torch.cuda.device_count() > 1:
            print("Using DataParallel with {} gpus".format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model = model.cuda(device=device)


    model.eval()
    torch.set_grad_enabled(False)
    ### Exract for train split ###
    mean, std = get_mean_std(1, dataset=args.dataset)
    normalize = get_normalize_method(mean, std, False, False)


    if split=='train':
        train_transforms = transforms.Compose([
            transforms.Resize((128, 171)),
            transforms.CenterCrop(128), #EDIT: 112
            transforms.ToTensor(),
            normalize
        ])
        if args.dataset == 'ucf101':
            train_dataset = UCF101ClipRetrievalDataset('/media/diskstation/datasets/UCF101', args.cl, 10, True, train_transforms)
        elif args.dataset == 'hmdb51':
            train_dataset = HMDB51ClipRetrievalDataset('/media/diskstation/datasets/HMDB51', args.cl, 10, True, train_transforms)

        train_dataloader = DataLoader(train_dataset, batch_size=args.bs, shuffle=False,
                                        num_workers=args.workers,
                                        pin_memory=True, drop_last=True)
        
        features = []
        classes = []
        for data in tqdm(train_dataloader):
            sampled_clips, idxs = data
            clips = sampled_clips.reshape((-1, 3, args.cl, 128, 128))
            inputs = clips.to(device)
            # forward
            if args.model=='CoCLR':
                outputs, _ = model(inputs)
            else:
                outputs = model(inputs)
            # print(outputs.shape)
            # exit()
            features.append(outputs.cpu().numpy().tolist())
            classes.append(idxs.cpu().numpy().tolist())

        features = np.array(features).reshape(-1, 10, outputs.shape[1])
        classes = np.array(classes).reshape(-1, 10)
        np.save(os.path.join(args.feature_dir, 'train_feature.npy'), features)
        np.save(os.path.join(args.feature_dir, 'train_class.npy'), classes)

    ### Exract for test split ###
    else:
        test_transforms = transforms.Compose([
            transforms.Resize((128, 171)),
            transforms.CenterCrop(128),
            transforms.ToTensor(),
            normalize
        ])
        if args.dataset == 'ucf101':
            test_dataset = UCF101ClipRetrievalDataset('/media/diskstation/datasets/UCF101', args.cl, 10, False, test_transforms)
        elif args.dataset == 'hmdb51':
            test_dataset = HMDB51ClipRetrievalDataset('/media/diskstation/datasets/HMDB51', args.cl, 10, False, test_transforms)
        test_dataloader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False,
                                        num_workers=args.workers,
                                        pin_memory=True, drop_last=True)

        features = []
        classes = []
        for data in tqdm(test_dataloader):
            sampled_clips, idxs = data
            clips = sampled_clips.reshape((-1, 3, args.cl, 128, 128))
            inputs = clips.to(device)
            # forward
            # outputs = model(inputs)
            if args.model=='CoCLR':
                outputs, _ = model(inputs)
            else:
                outputs = model(inputs)
            features.append(outputs.cpu().numpy().tolist())
            classes.append(idxs.cpu().numpy().tolist())

        features = np.array(features).reshape(-1, 10, outputs.shape[1])
        classes = np.array(classes).reshape(-1, 10)
        np.save(os.path.join(args.feature_dir, 'test_feature.npy'), features)
        np.save(os.path.join(args.feature_dir, 'test_class.npy'), classes)


def topk_retrieval(args):
    """Extract features from test split and search on train split features."""
    print('Load local .npy files.')
    X_train = np.load(os.path.join(args.feature_dir, 'train_feature.npy'))
    y_train = np.load(os.path.join(args.feature_dir, 'train_class.npy'))
    X_train = np.mean(X_train,1)
    y_train = y_train[:,0]
    X_train = X_train.reshape((-1, X_train.shape[-1]))
    y_train = y_train.reshape(-1)

    X_test = np.load(os.path.join(args.feature_dir, 'test_feature.npy'))
    y_test = np.load(os.path.join(args.feature_dir, 'test_class.npy'))
    X_test = np.mean(X_test,1)
    y_test = y_test[:,0]
    X_test = X_test.reshape((-1, X_test.shape[-1]))
    y_test = y_test.reshape(-1)

    ks = [1, 5, 10, 20, 50]
    topk_correct = {k:0 for k in ks}

    distances = cosine_distances(X_test, X_train)
    indices = np.argsort(distances)

    for k in ks:
        # print(k)
        top_k_indices = indices[:, :k]
        # print(top_k_indices.shape, y_test.shape)
        for ind, test_label in zip(top_k_indices, y_test):
            labels = y_train[ind]
            if test_label in labels:
                # print(test_label, labels)
                topk_correct[k] += 1

    for k in ks:
        correct = topk_correct[k]
        total = len(X_test)
        print('Top-{}, correct = {:.2f}, total = {}, acc = {:.3f}'.format(k, correct, total, correct/total))

    with open(os.path.join(args.feature_dir, 'topk_correct.json'), 'w') as fp:
        json.dump(topk_correct, fp)


def parse_args():
    parser = argparse.ArgumentParser(description='Frame Retrieval Experiment')
    parser.add_argument('--cl', type=int, default=16, help='clip length')
    # parser.add_argument('--model', type=str, default='c3d', help='c3d/r3d/r21d')
    parser.add_argument("--cfg", '-cfg', default=None, dest="cfg_file", type=str, help="Path to the config file")
    parser.add_argument('--dataset', type=str, default='ucf101', help='ucf101/hmdb51')
    parser.add_argument('--feature_dir', type=str, default='features', help='dir to store feature.npy')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id')
    parser.add_argument('--checkpoint_path', type=str, help='checkpoint path')
    parser.add_argument('--bs', type=int, default=16, help='mini-batch size')
    parser.add_argument('--workers', type=int, default=2, help='number of data loading workers')
    parser.add_argument('--model', type=str, default='ICVR', help='mainly used to differentiate CoCLR model from our model!')
    args = parser.parse_args()
    args.feature_dir = os.path.join(args.feature_dir, args.dataset)
    return args


if __name__ == '__main__':
    args = parse_args()
    print(vars(args))

    # if not os.path.exists(args.feature_dir):
    #     os.makedirs(args.feature_dir)
    #     extract_feature(args)

    if not os.path.exists(args.feature_dir):
        os.makedirs(args.feature_dir)
    
    if not os.path.exists(os.path.join(args.feature_dir,'train_feature.npy')):
        print('==> extracting train features')
        extract_feature(args, split='train')
    else:
        print('==> train features aready exist')
    if not os.path.exists(os.path.join(args.feature_dir,'test_feature.npy')):
        print('==> extracting test features')
        extract_feature(args, split='test')
    
    topk_retrieval(args)
