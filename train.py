import os
import torch
from torch import nn
import torch.optim as optim
from models.resnet import generate_model
from models.triplet_net import Tripletnet
from datasets import data_loader
import torch.backends.cudnn as cudnn


cudnn.benchmark = True

model_depth=18
n_classes=1039
n_input_channels=3
resnet_shortcut = 'B'
conv1_t_size = 7 #kernel size in t dim of conv1
conv1_t_stride = 1 #stride in t dim of conv1
no_max_pool = True #max pooling after conv1 is removed
resnet_widen_factor = 1 #number of feature maps of resnet is multiplied by this value
log_interval = 1 #log interval for batch number

cuda = False
if torch.cuda.is_available():
    print('cuda is ready')
    cuda = True
os.environ["CUDA_VISIBLE_DEVICES"]=str('0,1')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
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
        if batch_idx>3:
            break
        print('batch index:{}'.format(batch_idx))
        anchor, positive, negative = inputs
        anchor_target, positive_target, negative_target = targets
        print(anchor.size(), positive.size(), negative.size())

        if cuda:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
        dista, distb, embedded_x, embedded_y, embedded_z = tripletnet(anchor, positive, negative)
        print('dista', dista)
        print('distb', distb)
        # print('embedded_x:{}, embedded_y:{}, embedded_z:{}'.format(embedded_x.size(), embedded_y.size(), embedded_z.size()))

        #1 means, dista should be larger than distb
        target = torch.FloatTensor(dista.size()).fill_(-1)
        if cuda:
            target = target.to(device)


        loss_triplet = criterion(dista, distb, target)
        print('loss_triplet', loss_triplet)
        loss_embedd = embedded_x.norm(2) + embedded_y.norm(2) + embedded_z.norm(2)
        loss = loss_triplet + 0.001 *loss_embedd
        print(loss_embedd)
        #measure accuracy and record loss
        acc = accuracy(dista, distb)
        losses.update(loss_triplet, anchor.size(0))
        accs.update(acc, anchor.size(0))
        emb_norms.update(loss_embedd/3, anchor.size(0))

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{}]\t'
                  'Loss: {:.4f} ({:.4f}) \t'
                  'Acc: {:.2f}% ({:.2f}%) \t'
                  'Emb_Norm: {:.2f} ({:.2f})'.format(
                epoch, batch_idx * len(anchor), len(train_loader.dataset),
                losses.val, losses.avg,
                100. * accs.val, 100. * accs.avg, emb_norms.val, emb_norms.avg))


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
    pred = (dista - distb - margin).cpu().data
    return (pred > 0).sum()*1.0/dista.size()[0]



if __name__ == '__main__':
    pretrain_path = '/home/sherry/pretrained/r3d18_KM_200ep.pth'
    margin = 0.2
    lr = 0.01
    momentum=0.5
    epochs=1
    model=generate_model(model_depth=model_depth, n_classes=n_classes,
                        n_input_channels=n_input_channels, shortcut_type=resnet_shortcut,
                        conv1_t_size=conv1_t_size,
                        conv1_t_stride=conv1_t_stride,
                        no_max_pool=no_max_pool,
                        widen_factor=resnet_widen_factor)
    print('finished generating model...')
    model = load_pretrained_model(model, pretrain_path)
    tripletnet = Tripletnet(model)
    if cuda:
        if torch.cuda.device_count() > 1:
            print("Let's use {} GPUs".format(torch.cuda.device_count()))
            tripletnet = nn.DataParallel(tripletnet)
        tripletnet.to(device)

    train_data, train_loader = data_loader.get_train_data()
    # del train_data
    criterion = torch.nn.MarginRankingLoss(margin=margin)
    # criterion = criterion.to(device)# EDIT????
    optimizer = optim.SGD(tripletnet.parameters(), lr=lr, momentum=momentum)

    n_parameters = sum([p.data.nelement() for p in tripletnet.parameters()])
    print(' + Number of params: {}'.format(n_parameters))

    for epoch in range(1, epochs+1):
        train(train_loader, tripletnet, criterion, optimizer, epoch)
