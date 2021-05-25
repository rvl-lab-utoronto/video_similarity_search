"""Train 3D ConvNets to action classification."""
import os
import argparse
import time
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import torch.optim as optim
from tensorboardX import SummaryWriter

from iic_datasets.ucf101 import UCF101Dataset
from iic_datasets.hmdb51 import HMDB51Dataset

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

import pp


from datasets.spatial_transforms import (RandomResizedCrop, RandomHorizontalFlip,
                                ToTensor, ColorJitter, ColorDrop, GaussianBlur)
from datasets.data_loader import get_mean_std, get_normalize_method

from models.model_utils import (model_selector, multipathway_input,
                            load_pretrained_model, save_checkpoint, load_checkpoint,
                            AverageMeter, accuracy, create_output_dirs)
from config.m_parser import load_config, arg_parser
from config.default_params import get_cfg


def train(args, model, criterion, optimizer, device, train_dataloader, writer, epoch):
    torch.set_grad_enabled(True)
    model.train()

    running_loss = 0.0
    correct = 0
    for i, data in enumerate(train_dataloader, 1):
        # get inputs
        clips, idxs = data
        # print(clips.shape)
        inputs = clips.to(device)
        targets = idxs - 1
        targets = targets.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward and backward
        outputs = model(inputs) # return logits here
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # compute loss and acc
        running_loss += loss.item()
        pts = torch.argmax(outputs, dim=1)
        correct += torch.sum(targets == pts).item()
        # print statistics and write summary every N batch
        if i % args.pf == 0:
            avg_loss = running_loss / args.pf
            avg_acc = correct / (args.pf * args.batch_size)
            print('[TRAIN] epoch-{}, batch-{}, loss: {:.3f}, acc: {:.3f}'.format(epoch, i, avg_loss, avg_acc))
            step = (epoch-1)*len(train_dataloader) + i
            writer.add_scalar('train/CrossEntropyLoss', avg_loss, step)
            writer.add_scalar('train/Accuracy', avg_acc, step)
            running_loss = 0.0
            correct = 0
    # summary params and grads per eopch
    # for name, param in model.named_parameters():
        # writer.add_histogram('params/{}'.format(name), param, epoch)
        # writer.add_histogram('grads/{}'.format(name), param.grad, epoch)


def validate(args, model, criterion, device, val_dataloader, writer, epoch):
    torch.set_grad_enabled(False)
    model.eval()
    
    total_loss = 0.0
    correct = 0
    for i, data in enumerate(val_dataloader):
        # get inputs
        clips, idxs = data
        inputs = clips.to(device)

        targets = idxs - 1 #to avoid the error caused by label 101
        targets = targets.to(device)
        # forward
        outputs = model(inputs) # return logits here
        loss = criterion(outputs, targets)
        # compute loss and acc
        total_loss += loss.item()
        pts = torch.argmax(outputs, dim=1)
        correct += torch.sum(targets == pts).item()
        # print('correct: {}, {}, {}'.format(correct, targets, pts))
    avg_loss = total_loss / len(val_dataloader)
    avg_acc = correct / len(val_dataloader.dataset)
    writer.add_scalar('val/CrossEntropyLoss', avg_loss, epoch)
    writer.add_scalar('val/Accuracy', avg_acc, epoch)
    print('[VAL] loss: {:.3f}, acc: {:.3f}'.format(avg_loss, avg_acc))
    return avg_loss


def test_backup(args, model, criterion, device, test_dataloader):
    torch.set_grad_enabled(False)
    model.eval()

    total_loss = 0.0
    correct = 0
    for i, data in enumerate(test_dataloader, 1):
        # get inputs
        clips, idxs = data
        inputs = clips.to(device)
        targets = idxs.to(device)
        # forward
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        # compute loss and acc
        total_loss += loss.item()
        pts = torch.argmax(outputs, dim=1)
        correct += torch.sum(targets == pts).item()
        # print('correct: {}, {}, {}'.format(correct, targets, pts))
    avg_loss = total_loss / len(test_dataloader)
    avg_acc = correct / len(test_dataloader.dataset)
    print('[TEST] loss: {:.3f}, acc: {:.3f}'.format(avg_loss, avg_acc))
    return avg_loss


def test(args, model, criterion, device, test_dataloader, stdout=True):
    torch.set_grad_enabled(False)
    model.eval()

    total_loss = 0.0
    correct = 0
    confusion_matrix = torch.zeros((101, 101))
    # print(confusion_matrix.shape)
    for i, data in enumerate(test_dataloader, 1):
        # get inputs

        sampled_clips, idxs = data
        targets = idxs - 1
        targets = targets.to(device)
        outputs = []

        # stack, forward pass, and divide

        batch_size = sampled_clips.shape[0]
        num_test_samples = sampled_clips.shape[1]

        sampled_clips = sampled_clips.reshape((sampled_clips.shape[0]*sampled_clips.shape[1],sampled_clips.shape[2],
            sampled_clips.shape[3], sampled_clips.shape[4], sampled_clips.shape[5]))

        if i == 1:
            print(sampled_clips.shape)

        stacked_inputs = sampled_clips.to(device)
        stacked_o = model(stacked_inputs)
        for j in range(batch_size):
            o = stacked_o[j*num_test_samples:(j+1)*num_test_samples]
            o = torch.mean(o, dim=0)
            outputs.append(o)

        #for clips in sampled_clips:
        #    inputs = clips.to(device)
        #    # forward
        #    #print(inputs.shape)
        #    o = model(inputs)
        #    o = torch.mean(o, dim=0)
        #    # exit()
        #    outputs.append(o)

        outputs = torch.stack(outputs)
        loss = criterion(outputs, targets)
        # compute loss and acc
        total_loss += loss.item()
        pts = torch.argmax(outputs, dim=1)
        correct += torch.sum(targets == pts).item()

        # (unique, count) = torch.unique(pts, return_counts=True)
        # for i in range(len(unique)):
        #     confusion_matrix[targets[0], unique[i]] += count[i]
        for j in range(targets.shape[0]):
            confusion_matrix[targets[j], pts[j]] += 1

        # print(confusion_matrix[targets, pts])
        #if stdout: print('{}/{} | correct: {}, {}, {}'.format(i,
        #    len(test_dataloader), correct, targets, pts))
        if stdout: print('{}/{} | correct: {}'.format(i,
            len(test_dataloader), correct))
    
    #plot confusion matrix
    print(confusion_matrix)
    confusion_matrix = confusion_matrix.numpy()
    df_cm = pd.DataFrame(confusion_matrix, index = [i for i in range(0, 101)],
                  columns = [i for i in range(0, 101)])
    df_cm.to_csv("confusion_matrix.csv")

    # plt.figure()
    # sn.heatmap(df_cm, annot=True)
    # plt.savefig('confusion_matrix.png')

    avg_loss = total_loss / len(test_dataloader)
    avg_acc = correct / len(test_dataloader.dataset)
    print('[TEST] loss: {:.3f}, acc: {:.3f}'.format(avg_loss, avg_acc))
    return avg_loss, avg_acc

def run_test(args, model, device, stdout=True):
    test_transforms = transforms.Compose([
            transforms.Resize((128, 171)),
            transforms.CenterCrop(128),
            transforms.ToTensor(),
            normalize
        ])

    if args.dataset == 'ucf101':
        test_dataset = UCF101Dataset(os.path.join(args.dataset_root, 'UCF101'),
                args.cl, args.split, False, test_transforms,
                test_sample_num=args.test_sample_num) #10
    elif args.dataset == 'hmdb51':
        test_dataset = HMDB51Dataset(os.path.join(args.dataset_root, 'HMDB51'),
                args.cl, args.split, False, test_transforms, args.test_sample_num)

    test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)
    print('TEST video number: {}.'.format(len(test_dataset)))
    criterion = nn.CrossEntropyLoss()
    avg_loss, avg_acc = test(args, model, criterion, device, test_dataloader, stdout=stdout) #EDIT
    # test_backup(args, model, criterion, device, test_dataloader)
    return avg_loss, avg_acc

def parse_args():
    parser = argparse.ArgumentParser(description='Video Classification')
    parser.add_argument('--mode', type=str, default='train', help='train/test')
    parser.add_argument('--cfg', type=str, default=None, dest="cfg_file", help='config file')
    parser.add_argument('--model', type=str, default='resnet', help='c3d/r3d/r21d')
    parser.add_argument('--dataset', type=str, default='ucf101', help='ucf101/hmdb51')
    parser.add_argument('--split', type=str, default='1', help='dataset split')
    parser.add_argument('--cl', type=int, default=16, help='clip length')
    parser.add_argument('--gpu', type=int, default=1, help='GPU id')
    parser.add_argument('--dropout', default=0.9, type=float, help='dropout')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--momentum', type=float, default=9e-1, help='momentum')
    parser.add_argument('--wd', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--log', type=str, help='log directory')
    parser.add_argument('--checkpoint_path', type=str, help='checkpoint path')
    parser.add_argument('--desp', type=str, help='additional description')
    parser.add_argument('--epochs', type=int, default=150, help='number of total epochs to run')
    parser.add_argument('--start-epoch', type=int, default=1, help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch_size', type=int, default=8, help='mini-batch size')
    parser.add_argument('--workers', type=int, default=2, help='number of data loading workers')
    parser.add_argument('--pf', type=int, default=10, help='print frequency every batch')
    parser.add_argument('--seed', type=int, default=632, help='seed for initializing training.')
    parser.add_argument('--top_k', type=int, default=5, help='plot top k classes in confusion matrix')
    parser.add_argument('--test_sample_num', type=int, default=10, help='number of clips to sample per video in test mode')
    parser.add_argument('--test_batch_size', type=int, default=8, help='mini-batch size')
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="See config/defaults.py for all options",
    )
    parser.add_argument('--dataset_root', type=str, default='/media/diskstation/datasets', help='ucf/hmdb dataset root path')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print(vars(args))

    torch.backends.cudnn.benchmark = True
    # Force the pytorch to create context on the specific device 
    print('GPU:', args.gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    # device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    device = torch.cuda.current_device()
    print('device:', device)

    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.gpu:
            torch.cuda.manual_seed_all(args.seed)

    ########### model ##############
    if args.dataset == 'ucf101':
        class_num = 101
    elif args.dataset == 'hmdb51':
        class_num = 51

    cfg = get_cfg()
    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)
    model=model_selector(cfg, projection_head=False, classifier=True, dropout=args.dropout, 
            num_classes=class_num)

    # define normalize (used for train/test split)
    mean, std = get_mean_std(1, dataset=cfg.TRAIN.DATASET)
    normalize = get_normalize_method(mean, std, False, False, num_channels=cfg.DATA.INPUT_CHANNEL_NUM)

    if args.mode == 'train':  ########### Train #############
        if args.checkpoint_path:  # resume training
            try:
                model.load_state_dict(torch.load(args.checkpoint_path))
            except Exception as e:
                print("retry model loading with load_checkpoint()")
                start_epoch, best_acc = load_checkpoint(model, args.checkpoint_path, classifier=True)
                print("start_eppch:{}, best_acc:{}".format(start_epoch, best_acc))

            log_dir = os.path.dirname(args.checkpoint_path)

            

        else:
            if args.desp:
                exp_name = '{}_cl{}_{}_{}'.format(args.model, args.cl, args.desp, time.strftime('%m%d%H%M'))
            else:
                exp_name = '{}_cl{}_{}'.format(args.model, args.cl, time.strftime('%m%d%H%M'))
            log_dir = os.path.join(args.log, exp_name)

        writer = SummaryWriter(log_dir)

        model = model.cuda(device=device)


        # train_transforms = transforms.Compose([
        #     transforms.Resize((128, 171)),
        #     transforms.RandomCrop(128),
        #     transforms.ToTensor()
        # ])


        def get_spatial_transforms(normalize=normalize):
            print("==> Applying augmentation & normalization")
            spatial_transform = []
            spatial_transform.append(
                RandomResizedCrop(cfg.DATA.SAMPLE_SIZE, (0.25, 1.0), (0.75, 1.0/0.75))
            )
            spatial_transform.append(RandomHorizontalFlip(p=0.5))
            spatial_transform.append(ColorJitter(brightness=0.5, contrast=0.5,
                                                saturation=0.5, hue=0.5, p=0.8))
            spatial_transform.append(ColorDrop(p=0.2))
            # spatial_transform.append(GaussianBlur(p=0.2))
            spatial_transform.append(ToTensor())
            spatial_transform.append(normalize)
            return spatial_transform

        train_transforms = transforms.Compose(get_spatial_transforms())


        if args.dataset == 'ucf101':
            train_dataset = UCF101Dataset(os.path.join(args.dataset_root, 'UCF101'), args.cl, args.split, True, train_transforms)
            #val_size = 800
            #train_dataset, val_dataset = random_split(train_dataset, (len(train_dataset)-val_size, val_size))
        elif args.dataset == 'hmdb51':
            train_dataset = HMDB51Dataset(os.path.join(args.dataset_root, 'HMDB51'), args.cl, args.split, True, train_transforms)
            #val_size = 400
            #train_dataset, val_dataset = random_split(train_dataset, (len(train_dataset)-val_size, val_size))

        print('TRAIN video number: {}'.format(len(train_dataset)))
        #print('TRAIN video number: {}, VAL video number: {}.'.format(len(train_dataset), len(val_dataset)))
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                    num_workers=args.workers, pin_memory=True)
        #val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
        #                            num_workers=args.workers, pin_memory=True)

        if args.checkpoint_path:
            pass
        else:
            # save graph and clips_order samples
            for data in train_dataloader:
                clips, idxs = data
                writer.add_video('train/clips', clips, 0, fps=8)
                writer.add_text('train/idxs', str(idxs.tolist()), 0)
                clips = clips.to(device)
                writer.add_graph(model, clips)
                break
            # save init params at step 0
            # for name, param in model.named_parameters():
            #     writer.add_histogram('params/{}'.format(name), param, 0)

        ### loss funciton, optimizer and scheduler ###
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', min_lr=1e-5, patience=50, factor=0.1)

        prev_best_val_acc = -float('inf')
        prev_best_model_path = None
        for epoch in range(args.start_epoch, args.start_epoch+args.epochs):
            time_start = time.time()
            train(args, model, criterion, optimizer, device, train_dataloader, writer, epoch)
            print('Epoch time: {:.2f} hr.'.format((time.time() - time_start)/3600))
            #val_loss = validate(args, model, criterion, device, val_dataloader, writer, epoch)
            # scheduler.step(val_loss)         
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)

            # save checkpoint
            checkpoint_state = {
                'epoch': epoch+1,
                'state_dict': model.state_dict(),
                'best_prec1': prev_best_val_acc,
            }
            checkpoint_path = os.path.join(log_dir, 'checkpoint.pt')
            torch.save(checkpoint_state, checkpoint_path)
            print('Saved checkpoint to', checkpoint_path)

            if epoch % 10 == 0:
                torch.save(model.state_dict(), os.path.join(log_dir, 'model_{}.pt'.format(epoch)))
                avg_loss, avg_acc = run_test(args, model, device, stdout=False)

                if avg_acc > prev_best_val_acc:
                    model_path = os.path.join(log_dir, 'best_model_{}.pt'.format(epoch))
                    torch.save(model.state_dict(), model_path)
                    print('saved best model to', model_path)
                    prev_best_val_acc = avg_acc
                    if prev_best_model_path:
                        os.remove(prev_best_model_path)
                    prev_best_model_path = model_path

            # save model for the best val
            #if val_loss < prev_best_val_loss:
            #    model_path = os.path.join(log_dir, 'best_model_{}.pt'.format(epoch))
            #    torch.save(model.state_dict(), model_path)
            #    prev_best_val_loss = val_loss
            #    if prev_best_model_path:
            #        os.remove(prev_best_model_path)
            #    prev_best_model_path = model_path

    elif args.mode == 'test':  ########### Test #############
        model.load_state_dict(torch.load(args.checkpoint_path))
        model = model.cuda(device=device)


        avg_loss, avg_acc = run_test(args, model, device)
        # test_transforms = transforms.Compose([
        #     transforms.Resize((128, 171)),
        #     transforms.CenterCrop(128),
        #     transforms.ToTensor(),
        #     normalize
        # ])

        # if args.dataset == 'ucf101':
        #     test_dataset = UCF101Dataset('/media/diskstation/datasets/UCF101', args.cl, args.split, False, test_transforms, test_sample_num=10) #10
        # elif args.dataset == 'hmdb51':
        #     test_dataset = HMDB51Dataset('/media/diskstation/datasets/HMDB51', args.cl, args.split, False, test_transforms, 10)

        # test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
        #                         num_workers=args.workers, pin_memory=True)
        # print('TEST video number: {}.'.format(len(test_dataset)))
        # criterion = nn.CrossEntropyLoss()
        # test(args, model, criterion, device, test_dataloader) #EDIT
        # test_backup(args, model, criterion, device, test_dataloader)

    elif args.mode == 'plot':


        if args.dataset == 'ucf101':
            test_dataset = UCF101Dataset(os.path.join(args.dataset_root, 'UCF101'),
                    args.cl, args.split, False,
                    test_sample_num=args.test_sample_num) #10
        elif args.dataset == 'hmdb51':
            test_dataset = HMDB51Dataset(os.path.join(args.dataset_root,'HMDB51'),
                    args.cl, args.split, False, args.test_sample_num)


        confusion_matrix = pd.read_csv("confusion_matrix.csv")
        confusion_matrix = confusion_matrix.to_numpy()[:,1:]

        acc = np.diagonal(confusion_matrix)/np.sum(confusion_matrix, axis=1)
        top_k = acc.argsort()[:args.top_k]

        print(top_k)
        to_print = [(test_dataset.class_idx2label[k+1], acc[k]) for k in top_k]
        pp(to_print)
        idx = top_k
        for k in top_k:
            idx = np.concatenate((idx, confusion_matrix[k,:].argsort()[::-1][:4]))
        idx = np.unique(idx)

        to_select = np.ix_(top_k, idx)
        sub_matrix = confusion_matrix[to_select]
        sub_matrix = pd.DataFrame(sub_matrix)

        header = [x for x in test_dataset.class_idx2label[idx+1]]
        df_cm = pd.DataFrame(sub_matrix)
        df_cm.columns = header
        df_cm['classes'] = [x for x in test_dataset.class_idx2label[top_k+1]]
        df_cm.set_index("classes", inplace=True)
        # print(df_cm)
        plt.figure(figsize = (19,10))
        sn.heatmap(df_cm, annot=True)
        plt.savefig('confusion_matrix.png')
