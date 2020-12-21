"""
Created by Sherry Chen on Jul 14, 2020
retrieve the most similar clips
"""
import os
import argparse
import pickle as pkl
import pprint
import time
import numpy as np
import torch
import cv2
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
from datasets import data_loader
from models.triplet_net import Tripletnet
from models.model_utils import model_selector, multipathway_input, load_checkpoint, load_pretrained_model
from datasets.data_loader import build_spatial_transformation
from datasets.temporal_transforms import TemporalCenterFrame, TemporalSpecificCrop
from datasets.temporal_transforms import Compose as TemporalCompose
import misc.distributed_helper as du_helper
from config.m_parser import load_config, arg_parser
from misc.upload_gdrive import GoogleDriveUploader
from models.infoNCE import select_backbone
# num_exemplar = 10
log_interval = 10
top_k = 5
exemplar_file = None
#exemplar_file = '/home/sherry/output/u_exemplar.txt'
# np.random.seed(1)


# Argument parser
def m_arg_parser(parser):
    parser.add_argument(
        '--root_dir',
        type=str,
        default='.'
    )
    parser.add_argument(
        '--name',
        type=str,
        default=None,
        help='Please specify the name (e.g. ResNet18_K, SlowFast_U): '
    )
    parser.add_argument(
        '--num_exemplar',
        type=int,
        default=None,
        help='Please specify number of exemplar videos: '
    )
    parser.add_argument(
        '--heatmap',
        action='store_true',
        help='Run temporal heatmap visualization'
    )
    parser.add_argument(
        "--ex_idx",
        default=None,
        type=int,
        help='Exemplar video dataset index for the temporal heat map'
    )
    parser.add_argument(
        "--test_idx",
        default=None,
        type=int,
        help='Test video dataset index for the temporal heat map'
    )
    parser.add_argument(
        "--seed",
        default=0,
        type=int,
        help='seed for np.random'
    )
    parser.add_argument(
        "--load_pkl",
        action='store_true',
        help='load computed embeddings from the pickle file'
    )
    return parser





# def test_retrieval(model, criterion, transforms_cuda, device, epoch, args):
#     accuracy = [AverageMeter(),AverageMeter(),AverageMeter(),AverageMeter()]
#     model.eval()
    
#     def tr(x):
#         seq_len = 16
#         num_seq = 2
#         B = x.size(0); assert B == 1
#         test_sample = x.size(2)//(args.seq_len*args.num_seq)
#         return x.view(3,test_sample,args.num_seq,args.seq_len,args.img_dim,args.img_dim).permute(1,2,0,3,4,5)

#     with torch.no_grad():
#         # transform = transforms.Compose([
#         #             A.CenterCrop(size=(224,224)),
#         #             A.Scale(size=(args.img_dim,args.img_dim)),
#         #             A.ColorJitter(0.2, 0.2, 0.2, 0.1, p=0.3, consistent=True),
#         #             A.ToTensor()])

#         # if args.dataset == 'ucf101':
#         #     d_class = UCF101LMDB
#         # elif args.dataset == 'ucf101-f':
#         #     d_class = UCF101Flow_LMDB
#         # elif args.dataset == 'hmdb51':
#         #     d_class = HMDB51LMDB
#         # elif args.dataset == 'hmdb51-f':
#         #     d_class = HMDB51Flow_LMDB

#         # train_dataset = d_class(mode='train', 
#         #                     transform=transform, 
#         #                     num_frames=args.num_seq*args.seq_len,
#         #                     ds=args.ds,
#         #                     which_split=1,
#         #                     return_label=True,
#         #                     return_path=True)
#         # print('train dataset size: %d' % len(train_dataset))

#         # test_dataset = d_class(mode='test', 
#         #                     transform=transform, 
#         #                     num_frames=args.num_seq*args.seq_len,
#         #                     ds=args.ds,
#         #                     which_split=1,
#         #                     return_label=True,
#         #                     return_path=True)
#         # print('test dataset size: %d' % len(test_dataset))

#         # train_sampler = data.Sequential(train_dataset)
#         # test_sampler = data.Sequential(test_dataset)

#         # train_loader = data.DataLoader(train_dataset,
#         #                               batch_size=1,
#         #                               sampler=train_sampler,
#         #                               shuffle=False,
#         #                               num_workers=args.workers,
#         #                               pin_memory=True)
#         # test_loader = data.DataLoader(test_dataset,
#         #                               batch_size=1,
#         #                               sampler=test_sampler,
#         #                               shuffle=False,
#         #                               num_workers=args.workers,
#         #                               pin_memory=True)
#         if args.dirname is None:
#             dirname = 'feature'
#         else:
#             dirname = args.dirname

#         if os.path.exists(os.path.join(os.path.dirname(args.test), dirname, '%s_test_feature.pth.tar' % args.dataset)): 
#             test_feature = torch.load(os.path.join(os.path.dirname(args.test), dirname, '%s_test_feature.pth.tar' % args.dataset)).to(device)
#             test_label = torch.load(os.path.join(os.path.dirname(args.test), dirname, '%s_test_label.pth.tar' % args.dataset)).to(device)
#         else:
#             try: os.makedirs(os.path.join(os.path.dirname(args.test), dirname))
#             except: pass 

#             print('Computing test set feature ... ')
#             test_feature = None
#             test_label = []
#             test_vname = []
#             sample_id = 0 
#             for idx, (input_seq, target) in tqdm(enumerate(test_loader), total=len(test_loader)):
#                 B = 1
#                 input_seq = input_seq.to(device, non_blocking=True)
#                 input_seq = tr(input_seq)
#                 current_target, vname = target
#                 current_target = current_target.to(device, non_blocking=True)

#                 test_sample = input_seq.size(0)
#                 if args.other is not None: input_seq = input_seq.squeeze(1)
#                 logit, feature = model(input_seq)
#                 if test_feature is None:
#                     test_feature = torch.zeros(len(test_dataset), feature.size(-1), device=feature.device)

#                 if args.other is not None: 
#                     test_feature[sample_id,:] = feature.mean(0)
#                 else:
#                     test_feature[sample_id,:] = feature[:,-1,:].mean(0)
#                 test_label.append(current_target)
#                 test_vname.append(vname)
#                 sample_id += 1

#             print(test_feature.size())
#             # test_feature = torch.stack(test_feature, dim=0)
#             test_label = torch.cat(test_label, dim=0)
#             torch.save(test_feature, os.path.join(os.path.dirname(args.test), dirname, '%s_test_feature.pth.tar' % args.dataset))
#             torch.save(test_label, os.path.join(os.path.dirname(args.test), dirname, '%s_test_label.pth.tar' % args.dataset))
#             with open(os.path.join(os.path.dirname(args.test), dirname, '%s_test_vname.pkl' % args.dataset), 'wb') as fp:
#                 pickle.dump(test_vname, fp)


#         if os.path.exists(os.path.join(os.path.dirname(args.test), dirname, '%s_train_feature.pth.tar' % args.dataset)): 
#             train_feature = torch.load(os.path.join(os.path.dirname(args.test), dirname, '%s_train_feature.pth.tar' % args.dataset)).to(device)
#             train_label = torch.load(os.path.join(os.path.dirname(args.test), dirname, '%s_train_label.pth.tar' % args.dataset)).to(device)
#         else:
#             print('Computing train set feature ... ')
#             train_feature = None
#             train_label = []
#             train_vname = []
#             sample_id = 0
#             for idx, (input_seq, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
#                 B = 1
#                 input_seq = input_seq.to(device, non_blocking=True)
#                 input_seq = tr(input_seq)
#                 current_target, vname = target
#                 current_target = current_target.to(device, non_blocking=True)

#                 test_sample = input_seq.size(0)
#                 if args.other is not None: input_seq = input_seq.squeeze(1)
#                 logit, feature = model(input_seq)
#                 if train_feature is None:
#                     train_feature = torch.zeros(len(train_dataset), feature.size(-1), device=feature.device)

#                 if args.other is not None: 
#                     train_feature[sample_id,:] = feature.mean(0)
#                 else:
#                     train_feature[sample_id,:] = feature[:,-1,:].mean(0)
#                 # train_feature.append(feature[:,-1,:].mean(0))
#                 train_label.append(current_target)
#                 train_vname.append(vname)
#                 sample_id += 1
#             # train_feature = torch.stack(train_feature, dim=0)
#             print(train_feature.size())
#             train_label = torch.cat(train_label, dim=0)
#             torch.save(train_feature, os.path.join(os.path.dirname(args.test), dirname, '%s_train_feature.pth.tar' % args.dataset))
#             torch.save(train_label, os.path.join(os.path.dirname(args.test), dirname, '%s_train_label.pth.tar' % args.dataset))
#             with open(os.path.join(os.path.dirname(args.test), dirname, '%s_train_vname.pkl' % args.dataset), 'wb') as fp:
#                 pickle.dump(train_vname, fp)

#         ks = [1,5,10,20,50]
#         NN_acc = []

#         # centering
#         test_feature = test_feature - test_feature.mean(dim=0, keepdim=True)
#         train_feature = train_feature - train_feature.mean(dim=0, keepdim=True)

#         # normalize
#         test_feature = F.normalize(test_feature, p=2, dim=1)
#         train_feature = F.normalize(train_feature, p=2, dim=1)

#         # dot product
#         sim = test_feature.matmul(train_feature.t())

#         torch.save(sim, os.path.join(os.path.dirname(args.test), dirname, '%s_sim.pth.tar' % args.dataset))

#         for k in ks:
#             topkval, topkidx = torch.topk(sim, k, dim=1)
#             acc = torch.any(train_label[topkidx] == test_label.unsqueeze(1), dim=1).float().mean().item()
#             NN_acc.append(acc)
#             print('%dNN acc = %.4f' % (k, acc))

#         args.logger.log('NN-Retrieval on %s:' % args.dataset)
#         for k,acc in zip(ks, NN_acc):
#             args.logger.log('\t%dNN acc = %.4f' % (k, acc))

#         with open(os.path.join(os.path.dirname(args.test), dirname, '%s_test_vname.pkl' % args.dataset), 'rb') as fp:
#             test_vname = pickle.load(fp)

#         with open(os.path.join(os.path.dirname(args.test), dirname, '%s_train_vname.pkl' % args.dataset), 'rb') as fp:
#             train_vname = pickle.load(fp)

#         sys.exit(0)





class LinearClassifier(nn.Module):
    def __init__(self, num_class=101, 
                 network='resnet50', 
                 dropout=0.5, 
                 use_dropout=True, 
                 use_l2_norm=False,
                 use_final_bn=False):
        super(LinearClassifier, self).__init__()
        self.network = network
        self.num_class = num_class
        self.dropout = dropout
        self.use_dropout = use_dropout
        self.use_l2_norm = use_l2_norm
        self.use_final_bn = use_final_bn
        
        message = 'Classifier to %d classes with %s backbone;' % (num_class, network)
        if use_dropout: message += ' + dropout %f' % dropout
        if use_l2_norm: message += ' + L2Norm'
        if use_final_bn: message += ' + final BN'
        print(message)

        self.backbone, self.param = select_backbone(network)
        
        if use_final_bn:
            self.final_bn = nn.BatchNorm1d(self.param['feature_size'])
            self.final_bn.weight.data.fill_(1)
            self.final_bn.bias.data.zero_()
        
        if use_dropout:
            self.final_fc = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(self.param['feature_size'], self.num_class))
        else:
            self.final_fc = nn.Sequential(
                nn.Linear(self.param['feature_size'], self.num_class))
        self._initialize_weights(self.final_fc)
        
    def forward(self, block):
        # print(block)
        (B, C, T, H, W) = block.shape
        feat3d = self.backbone(block)
        feat3d = F.adaptive_avg_pool3d(feat3d, (1,1,1)) # [B,C,1,1,1]
        feat3d = feat3d.view(B, self.param['feature_size']) # [B,C]

        if self.use_l2_norm:
            feat3d = F.normalize(feat3d, p=2, dim=1)
        
        if self.use_final_bn:
            logit = self.final_fc(self.final_bn(feat3d))
        else:
            logit = self.final_fc(feat3d)

        return logit, feat3d

    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.normal_(param, 0.01)


def evaluate(cfg, model, cuda, device, data_loader, split='train', is_master_proc=True):
    #log_interval=len(data_loader.dataset)//5
    log_interval = 5

    model.eval()
    embedding = []
    # vid_info = []
    labels = []
    world_size = du_helper.get_world_size()
    with torch.no_grad():
        for batch_idx, (input, targets, info) in enumerate(data_loader):
            batch_size = input.size(0)
            if cfg.MODEL.ARCH == 'slowfast':
                input = multipathway_input(input, cfg)
                if cuda:
                    for i in range(len(input)):
                        input[i] = input[i].to(device)
            else:
                if cuda:
                    input= input.to(device)
            if cuda:
                targets = targets.to(device)

            _, embedd = model(input)
            if cfg.NUM_GPUS > 1:
                embedd, targets = du_helper.all_gather([embedd, targets])
            embedding.append(embedd.detach().cpu())
            labels.append(targets.detach().cpu())
            # vid_info.extend(info)
            # print('embedd size', embedd.size())
            batch_size_world = batch_size * world_size
            if ((batch_idx + 1) * world_size) % log_interval == 0 and is_master_proc:
                print('{} [{}/{} | {:.1f}%]'.format(split, (batch_idx+1)*batch_size_world, len(data_loader.dataset), 
                    ((batch_idx+1)*100.*batch_size_world/len(data_loader.dataset))))

    embeddings = torch.cat(embedding, dim=0)
    labels = torch.cat(labels, dim=0).tolist()
    #if is_master_proc: print('embeddings size', embeddings.size())
    return embeddings, labels


def get_distance_matrix(x_embeddings, y_embeddings=None, dist_metric='cosine'):

    #print('Dist metric:', dist_metric)
    assert(dist_metric in ['cosine', 'euclidean'])
    if dist_metric == 'cosine':
        distance_matrix = cosine_distances(x_embeddings, Y=y_embeddings)
    elif dist_metric == 'euclidean':
        distance_matrix = euclidean_distances(x_embeddings, Y=y_embeddings)
    print('Distance matrix shape:', distance_matrix.shape)

    if y_embeddings is None:
        np.fill_diagonal(distance_matrix, float('inf'))
    return distance_matrix



def get_closest_data_mat(distance_matrix, top_k):
    idx = np.argpartition(distance_matrix, top_k, axis=-1)
    distance_matrix_topk_unsorted = np.take_along_axis(distance_matrix, idx[:,:top_k], axis=-1)
    idx_sorted_indices = np.argsort(distance_matrix_topk_unsorted, axis=-1)
    top_k = np.take_along_axis(idx, idx_sorted_indices, axis=-1)
    return top_k  # dim: distance_matrix.shape[0] x top_k


def get_closest_data(distance_matrix, exemplar_idx, top_k):
    test_array = distance_matrix[exemplar_idx]
    idx = np.argpartition(test_array, top_k)
    top_k = idx[np.argsort(test_array[idx[:top_k]])]
    return top_k


def plot_img(cfg, fig, val_data, train_data, num_exemplar, row, exemplar_idx, k_idx, spatial_transform=None, temporal_transform=None, output=None):
    exemplar_frame = val_data._loading_img_path(exemplar_idx, temporal_transform)
    test_frame = [train_data._loading_img_path(i, temporal_transform) for i in k_idx]

    exemplar_title = '-'.join(exemplar_frame.split('/')[-3:-2])

    print(exemplar_frame)
    print('top k ids:', end=' ')
    for i in k_idx:
        print(i, end=' ')
    pprint.pprint(test_frame)

    ax = fig.add_subplot(num_exemplar,len(test_frame)+1, row*(len(test_frame)+1)+1)
    image = plt.imread(exemplar_frame)
    plt.imshow(image)
    ax.set_title(exemplar_title, fontsize=5, pad=0.3)
    plt.axis('off')
    for i in range(len(test_frame)):
        test_title = '-'.join(test_frame[i].split('/')[-3:-2])
        ax = fig.add_subplot(num_exemplar,len(test_frame)+1, row*(len(test_frame)+1)+i+2)
        image = plt.imread(test_frame[i])
        plt.imshow(image)
        ax.set_title(test_title, fontsize=5, pad=0.3)
        plt.axis('off')

    with open(os.path.join(output, 'results.txt'), 'a') as f:
        f.write('exemplar_frame:\n{}\n'.format(exemplar_frame))
        for frame in test_frame:
            f.write(frame)
            f.write('\n')
        f.write('\n')

    with open(os.path.join(output, 'exemplar.txt'), 'a') as f:
        f.write('{}, {}'.format(exemplar_idx, exemplar_frame))
        f.write('\n')


def load_exemplar(exemplar_file):
    with open(exemplar_file, 'r') as f:
        lines = f.readlines()
    exemplar_idx  = []
    for line in lines:
        exemplar_idx.append(int(line.split(',')[0].strip()))
    return exemplar_idx


def get_topk_acc(distance_matrix, x_labels, y_labels=None, top_ks = [1,5,10,20]):
    top_k = top_ks[-1] 
    topk_sum = 0
    topk_indices = get_closest_data_mat(distance_matrix, top_k=top_k)
    # print('topk_indices', topk_indices.size())
    if y_labels is None:
        y_labels = x_labels
    
    acc = []
    for i, x_label in enumerate(x_labels):
        cur_acc = []
        # print('x_label', x_label)
        for k in top_ks:
            topk_idx = topk_indices[:, :k]
            cur_topk_idx = topk_idx[i]
            topk_labels = [y_labels[j] for j in cur_topk_idx]
            topk_sum = int(x_label in topk_labels)
            cur_acc.append(topk_sum)
        acc.append(cur_acc)
    # print(acc.size())
    acc = np.mean(np.array(acc), axis=0)
    return acc

def get_embeddings_and_labels(args, cfg, model, cuda, device, data_loader,
        split='val', is_master_proc=True, load_pkl=False, save_pkl=True):

    if split == 'train':
        embeddings_pkl = os.path.join(cfg.OUTPUT_PATH, 'train_embeddings.pkl')
        labels_pkl = os.path.join(cfg.OUTPUT_PATH, 'train_labels.pkl')
    else:
        embeddings_pkl = os.path.join(cfg.OUTPUT_PATH, 'val_embeddings.pkl')
        labels_pkl = os.path.join(cfg.OUTPUT_PATH, 'val_labels.pkl')

    if os.path.exists(embeddings_pkl) and os.path.exists(labels_pkl) and load_pkl:
        with open(embeddings_pkl, 'rb') as handle:
            embeddings = torch.load(handle)
        with open(labels_pkl, 'rb') as handle:
            labels = torch.load(handle)
        print('retrieved {}_embeddings'.format(split), embeddings.size(), 'labels', len(labels))
    else:
        embeddings, labels = evaluate(cfg, model, cuda, device, data_loader, split=split, is_master_proc=is_master_proc)
        if save_pkl:
            with open(embeddings_pkl, 'wb') as handle:
                torch.save(embeddings, handle, pickle_protocol=pkl.HIGHEST_PROTOCOL)
            with open(labels_pkl, 'wb') as handle:
                torch.save(labels, handle, pickle_protocol=pkl.HIGHEST_PROTOCOL)

    return embeddings, labels


def k_nearest_embeddings(args, model, cuda, device, train_loader, test_loader, train_data, val_data, cfg, plot=True,
                        epoch=None, is_master_proc=True,
                        evaluate_output=None, num_exemplar=None, service=None,
                        load_pkl=False, out_filename='global_retrieval_acc'):
    print ('Getting embeddings...')
    val_embeddings, val_labels = get_embeddings_and_labels(args, cfg, model, cuda, device, test_loader,
                                                        split='val', is_master_proc=is_master_proc, load_pkl=load_pkl)
    train_embeddings, train_labels = get_embeddings_and_labels(args, cfg, model, cuda, device, train_loader,
                                                        split='train', is_master_proc=is_master_proc, load_pkl=load_pkl)
    acc = []

    print ('Computing top1/5/10/20 Acc...')
    if (is_master_proc):
        distance_matrix = get_distance_matrix(val_embeddings, None, dist_metric=cfg.LOSS.DIST_METRIC)
        acc = get_topk_acc(distance_matrix, val_labels, y_labels=None)
        if epoch is not None:
            to_write = 'epoch:{} {:.2f} {:.2f}'.format(epoch, 100.*acc[0], 100.*acc[1], 100.*acc[2], 100.*acc[3])
            msg = '\nTest Set: Top1 Acc: {:.2f}%, Top5 Acc: {:.2f}%, Top10 Acc: {:.2f}%, Top20 Acc: {:.2f}%'.format(100.*acc[0], 100.*acc[1], 100.*acc[2], 100.*acc[3])
            to_write += '\n'
            with open('{}/tnet_checkpoints/{}.txt'.format(cfg.OUTPUT_PATH, out_filename), "a") as val_file:
                val_file.write(to_write)

        if plot:
            spatial_transform = build_spatial_transformation(cfg, 'val')
            temporal_transform = [TemporalCenterFrame()]
            temporal_transform = TemporalCompose(temporal_transform)
            fig = plt.figure()
            for i in range(num_exemplar):
                exemplar_idx = np.random.randint(0, distance_matrix.shape[0]-1)
                print('exemplar video id: {}'.format(exemplar_idx))
                k_idx = get_closest_data(distance_matrix, exemplar_idx, top_k)
                print(k_idx, len(train_data))
                k_nearest_data = [train_data[i] for i in k_idx]
                plot_img(cfg, fig, val_data, train_data, num_exemplar, i, exemplar_idx, k_idx, spatial_transform, temporal_transform, output=evaluate_output)
            # plt.show()
            png_file = os.path.join(evaluate_output, '{}_plot.png'.format(os.path.basename(evaluate_output)))
            fig.tight_layout(pad=3.5)
            plt.savefig(png_file, dpi=300)
            service.upload_file_to_gdrive(png_file, 'evaluate')
            print('figure saved to: {}, and uploaded to GoogleDrive'.format(png_file))
        # print(acc)
        print('Top1 Acc: {:.2f}%, Top5 Acc: {:.2f}%, Top10 Acc: {:.2f}%, Top20 Acc: {:.2f}%'.format(100.*acc[0], 100.*acc[1], 100.*acc[2], 100.*acc[3]))
    return acc


def temporal_heat_map(model, data, cfg, evaluate_output, exemplar_idx=455,
        test_idx=456):

    num_frames_exemplar = data.data[exemplar_idx]['num_frames']

    exemplar_video_full, _, _ = data._get_video_custom_temporal(exemplar_idx)  # full size
    exemplar_video_full = exemplar_video_full.unsqueeze(0)

    num_frames_crop = cfg.DATA.SAMPLE_DURATION
    stride = num_frames_crop // 2
    dists = []

    model.eval()
    with torch.no_grad():
        test_video, _, _ = data.__getitem__(test_idx)  # cropped size
        test_video = test_video.unsqueeze(0)
        print('Test input size:', test_video.size(), '\n')
        if (cfg.MODEL.ARCH == 'slowfast'):
            test_video_in = multipathway_input(test_video, cfg)
            if cuda:
                for i in range(len(test_video_in)):
                    test_video_in[i] = test_video_in[i].to(device)
        else:
            if cuda:
                test_video_in = test_video_in.to(device)
        test_embedding = model(test_video_in)
        #print('Test embed size:', test_embedding.size())

        # Loop across exemplar video, use [i-cfg.DATA.SAMPLE_SIZE,...,i] as the frames for the temporal crop
        for i in range(num_frames_crop, num_frames_exemplar, stride):
            temporal_transform_exemplar = TemporalSpecificCrop(begin_index=i-num_frames_crop, size=num_frames_crop)
            exemplar_video, _, _ = data._get_video_custom_temporal(exemplar_idx, temporal_transform_exemplar)  # full siz
            exemplar_video = exemplar_video.unsqueeze(0)

            if (cfg.MODEL.ARCH == 'slowfast'):
                exemplar_video_in = multipathway_input(exemplar_video, cfg)
                if cuda:
                    for j in range(len(exemplar_video_in)):
                        exemplar_video_in[j] = exemplar_video_in[j].to(device)
            else:
                if cuda:
                    exemplar_video_in = exemplar_video_in.to(device)

            exemplar_embedding = model(exemplar_video_in)
            #print('Exemplar input size:', exemplar_video.size())
            #print('Exemplar embed size:', exemplar_embedding.size())
            dist = F.pairwise_distance(exemplar_embedding, test_embedding, 2)
            dists.append(dist.item())

    #print(dists)
    x = []
    y = []
    plt.show()
    axes = plt.gca()
    axes.set_xlim(0, num_frames_exemplar)
    axes.set_ylim(0, max(dists))
    line, = axes.plot(x, y, 'b-')
    dist_idx = 0

    # channels x frames x width, height --> frames x width x height x channels
    video_ex = exemplar_video_full[0].permute(1,2,3,0)
    video_test = test_video[0].permute(1,2,3,0)
    fps = 25.0
    for i in range(len(video_ex)):
        blank_divider = np.full((2,128,3), 256, dtype=int)
        np_vertical_stack = np.vstack(( video_ex[i].numpy(), blank_divider, video_test[i % len(video_test)].numpy() ))
        cv2.imshow('Videos', np_vertical_stack)

        # show plot of embedding distance for past num_frames_crop frames of exemplar video
        if i >= num_frames_crop and (i - num_frames_crop) % stride == 0:
            x.append(i)
            y.append(dists[dist_idx])
            line.set_xdata(x)
            line.set_ydata(y)
            plt.draw()
            plt.pause(1e-17)
            dist_idx += 1

        cv2.waitKey(int(1.0/fps*1000.0))


if __name__ == '__main__':
    args = m_arg_parser(arg_parser()).parse_args()
    cfg = load_config(args)

    print("Train batch size:", cfg.TRAIN.BATCH_SIZE)
    print("Val batch size:", cfg.VAL.BATCH_SIZE)

    np.random.seed(args.seed)

    force_data_parallel = True
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    global cuda; cuda = torch.cuda.is_available()
    global device; device = torch.device("cuda" if cuda else "cpu")

    name = args.name
    num_exemplar = args.num_exemplar

    if not name:
        name = input('Please specify the name (e.g. ResNet18_K, SlowFast_U): ')
    if not num_exemplar:
        num_exemplar = int(input('Please specify number of exemplar videos: '))

    if not cfg.OUTPUT_PATH:
        output = input('Please specify output directory: ')
        cfg.OUTPUT_PATH = output
    else:
        output = cfg.OUTPUT_PATH

    start = time.time()
    now = datetime.now()
    evaluate_output = os.path.join(output, '{}_evaluate'.format(name))
    if not os.path.exists(evaluate_output):
        os.makedirs(evaluate_output)
        print('made output dir:{}'.format(evaluate_output))


    # ============================== Model Setup ===============================
    # Check if this is the master process (true if not distributed)
    is_master_proc = du_helper.is_master_proc(cfg.NUM_GPUS)
    model = LinearClassifier(
                    network='s3d', 
                    num_class=101,
                    dropout=0.9,
                    use_dropout=False,
                    use_final_bn=False,
                    use_l2_norm=False)    # Select appropriate model

    # if(is_master_proc):
    #     print('\n==> Generating {} backbone model...'.format(cfg.MODEL.ARCH))
    # # model=model_selector(cfg, projection_head=False)
    # n_parameters = sum([p.data.nelement() for p in model.parameters()])
    # if(is_master_proc):
    #     print('Number of params: {}'.format(n_parameters))

    # Transfer model to DDP
    if cuda:
        if torch.cuda.device_count() > 1:
            print("Using DataParallel with {} gpus".format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model = model.cuda(device=device)

    # if args.pretrain_path is not None:
    #     model = load_pretrained_model(model, args.pretrain_path, is_master_proc)

    # Load similarity network checkpoint if path exists
    if args.checkpoint_path is not None:
        checkpoint = torch.load(args.checkpoint_path, map_location=torch.device('cpu'))
        epoch = checkpoint['epoch']
        state_dict = checkpoint['state_dict']

        # if args.retrieval_ucf or args.retrieval_full: # if directly test on pretrained network
        new_dict = {}
        for k,v in state_dict.items():
            if 'encoder_q.0.' in k:
                new_k = k.replace('encoder_q.0.', '')
                new_dict[new_k] = v 


            # k = k.replace('encoder_q.0.', 'backbone.')
            # new_dict[k] = v
        state_dict = new_dict
        model.backbone.load_state_dict(state_dict)
    # print(model)

    # print('=> finished generating similarity network...')

    # # ============================== Data Loaders ==============================

    train_loader, (train_data, _) = data_loader.build_data_loader('train', cfg, triplets=False)
    test_loader, (val_data, _) = data_loader.build_data_loader('val', cfg, triplets=False)#, val_sample=None)

    # # ================================ Evaluate ================================

    if args.heatmap:
        if args.ex_idx and args.test_idx:
            temporal_heat_map(model, data, cfg, evaluate_output, args.ex_idx,
                args.test_idx)
        else:
            print ('No exemplar and test indices provided')
            temporal_heat_map(model, data, cfg, evaluate_output)
    else:
        k_nearest_embeddings(args, model, cuda, device, train_loader, test_loader,
                        train_data, val_data, cfg, evaluate_output=evaluate_output,
                        num_exemplar=num_exemplar, service=GoogleDriveUploader(), load_pkl=args.load_pkl)
        print('total runtime: {}s'.format(time.time()-start))
