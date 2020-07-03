"""
Created by Sherry Chen on Jul 3, 2020
Build and Train Triplet network. Supports saving and loading checkpoints,
"""
import os
import argparse
import shutil
import tqdm
import torch
from torch import nn
import torch.optim as optim
from models.resnet import generate_model
from models.triplet_net import Tripletnet
from datasets import data_loader
import torch.backends.cudnn as cudnn

from pytorch_memlab import MemReporter

# cudnn.benchmark = True

def get_args():
    parser.argparse.ArgumentParser("3D ResNet TripletNet")
    # TODO: move all global variable to args
    pass


model_depth=18
n_classes=1039
n_input_channels=3
resnet_shortcut = 'B'
conv1_t_size = 7 #kernel size in t dim of conv1
conv1_t_stride = 1 #stride in t dim of conv1
no_max_pool = True #max pooling after conv1 is removed
resnet_widen_factor = 1 #number of feature maps of resnet is multiplied by this value
log_interval = 5 #log interval for batch number
root_dir = '/home/sherry/tnet_checkpoints/diff_instance'

# resume='/home/sherry/tnet_checkpoints/r3d18/model_best.pth.tar'
# pretrain = False
resume = None
pretrain = None

cuda = False
if torch.cuda.is_available():
    print('cuda is ready')
    cuda = True
os.environ["CUDA_VISIBLE_DEVICES"]=str('0,1')
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device('cuda:1')
# print(device)
# print(torch.cuda.device_count())



def load_pretrained_model(model, pretrain_path):
    if pretrain_path:
        print('loading pretrained model {}'.format(pretrain_path))
        pretrain = torch.load(pretrain_path, map_location='cpu')

        model.load_state_dict(pretrain['state_dict'])
        tmp_model = model

    return model


def train(train_loader, tripletnet, criterion, optimizer, epoch):
    losses = AverageMeter()
    accs = AverageMeter()
    emb_norms=AverageMeter()

    #switching to training mode
    tripletnet.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        anchor, positive, negative = inputs
        (anchor_target, positive_target, negative_target) = targets

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
        losses.update(loss_triplet.cpu(), anchor.size(0))
        accs.update(acc, anchor.size(0))
        emb_norms.update(loss_embedd.cpu()/3, anchor.size(0))

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
            print('Train Epoch: {} [{}/{}]\t'
                  'Loss: {:.4f} ({:.4f}) \t'
                  'Acc: {:.2f}% ({:.2f}%) \t'
                  'Emb_Norm: {:.2f} ({:.2f})'.format(
                epoch, batch_idx * len(anchor), len(train_loader.dataset),
                losses.val, losses.avg,
                100. * accs.val, 100. * accs.avg, emb_norms.val, emb_norms.avg))


def validate(val_loader, tripletnet, criterion, epoch):
    losses = AverageMeter()
    accs = AverageMeter()

    tripletnet.eval()
    torch.cuda.empty_cache()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            (anchor, positive, negative) = inputs
            (anchor_target, positive_target, negative_target) = targets
            if cuda:
                anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            dista, distb, _, _, _ = tripletnet(anchor, positive, negative)
            target = torch.FloatTensor(dista.size()).fill_(-1)
            if cuda:
                target = target.to(device)

            test_loss = criterion(dista, distb, target)

            # measure accuracy and record loss
            acc = accuracy(dista, distb)
            accs.update(acc.cpu(), anchor.size(0))
            losses.update(test_loss.cpu(), anchor.size(0))

            torch.cuda.empty_cache()

    print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
        losses.avg, 100. * accs.avg))
    return accs.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = "tnet_checkpoints/%s/"%(model_name)
    directory = os.path.join(root_dir, directory)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    print('=> checkpoint:{} saved...'.format(filename))
    if is_best:
        shutil.copyfile(filename,  os.path.join(directory, 'model_best.pth.tar'))
        print('=> best_model saved as:{}'.format(os.path.join(directory, 'model_best.pth.tar')))


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
    # pred = (dista - distb - margin).cpu().data
    pred = (distb - dista - margin).cpu().data
    # print('pred', pred)
    return (pred > 0).sum()*1.0/dista.size()[0]






if __name__ == '__main__':
    pretrain_path = '/home/sherry/pretrained/r3d18_KM_200ep.pth'
    margin = 0.2
    lr = 0.05
    momentum=0.5
    epochs=20
    best_acc = 0
    start_epoch = 0
    model_name = os.path.basename(pretrain_path).split('_')[0]
    cudnn.benchmark = True

    # torch.cuda.empty_cache()

    model=generate_model(model_depth=model_depth, n_classes=n_classes,
                        n_input_channels=n_input_channels, shortcut_type=resnet_shortcut,
                        conv1_t_size=conv1_t_size,
                        conv1_t_stride=conv1_t_stride,
                        no_max_pool=no_max_pool,
                        widen_factor=resnet_widen_factor)
    print('=> finished generating model...')
    if pretrain:
        print('=> loading pretrained model')
        model = load_pretrained_model(model, pretrain_path)
        print('=> pretrain model:{} is loaded'.format(pretrain_path))

    tripletnet = Tripletnet(model)

    if resume:
        if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume)
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            tripletnet.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                    .format(resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume))

    if cuda:
        # if torch.cuda.device_count() > 1:
            # print("Let's use {} GPUs".format(torch.cuda.device_count()))
            # tripletnet = nn.DataParallel(tripletnet, device_ids=[0, 1])
            # print('devices:{}'.format(tripletnet.device_ids))
        tripletnet.to(device)

    train_data, train_loader = data_loader.get_train_data()
    val_data, val_loader = data_loader.get_val_data()

    criterion = torch.nn.MarginRankingLoss(margin=margin).to(device)
    optimizer = optim.SGD(tripletnet.parameters(), lr=lr, momentum=momentum)

    n_parameters = sum([p.data.nelement() for p in tripletnet.parameters()])
    print(' + Number of params: {}'.format(n_parameters))


    for epoch in range(start_epoch, epochs+1):
        train(train_loader, tripletnet, criterion, optimizer, epoch)
        acc = validate(val_loader, tripletnet, criterion, epoch)
        print('epoch:{}, acc:{}'.format(epoch, acc))
        is_best = acc > best_acc

        best_acc = max(acc, best_acc)
        save_checkpoint({
            'epoch': epoch+1,
            'state_dict':tripletnet.state_dict(),
            'best_prec1': best_acc,
        }, is_best)
