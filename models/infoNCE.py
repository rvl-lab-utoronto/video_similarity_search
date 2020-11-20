# MoCo-related code is modified from https://github.com/facebookresearch/moco
import sys
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
# from S3D_network import S3D
# sys.path.append('.')
# from backbone.select_backbone import select_backbone


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def select_backbone(network, first_channel=3):
    param = {'feature_size': 1024}
    if network == 's3d':
        model = S3D(input_channel=first_channel)
    elif network == 's3dg':
        model = S3D(input_channel=first_channel, gating=True)
    else: 
        raise NotImplementedError

    return model, param

class InfoNCE(nn.Module):
    '''
    Basically, it's a MoCo for video input: https://arxiv.org/abs/1911.05722
    '''
    def __init__(self, network='s3d', dim=128, K=2048, m=0.999, T=0.07):
        '''
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 2048)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        '''
        super(InfoNCE, self).__init__()

        self.dim = dim 
        self.K = K
        self.m = m
        self.T = T

        # create the encoders (including non-linear projection head: 2 FC layers)
        backbone, self.param = select_backbone(network)
        feature_size = self.param['feature_size']
        self.encoder_q = nn.Sequential(
                            backbone, 
                            nn.AdaptiveAvgPool3d((1,1,1)),
                            nn.Conv3d(feature_size, feature_size, kernel_size=1, bias=True),
                            nn.ReLU(),
                            nn.Conv3d(feature_size, dim, kernel_size=1, bias=True))

        backbone, _ = select_backbone(network)
        self.encoder_k = nn.Sequential(
                            backbone, 
                            nn.AdaptiveAvgPool3d((1,1,1)),
                            nn.Conv3d(feature_size, feature_size, kernel_size=1, bias=True),
                            nn.ReLU(),
                            nn.Conv3d(feature_size, dim, kernel_size=1, bias=True))

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # Notes: for handling sibling videos, e.g. for UCF101 dataset


    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        '''Momentum update of the key encoder'''
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        # keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        '''
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        '''
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        '''
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        '''
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, block):
        '''Output: logits, targets'''
        (B, N, *_) = block.shape # [B,N,C,T,H,W]
        assert N == 2
        x1 = block[:,0,:].contiguous()
        x2 = block[:,1,:].contiguous()

        # compute query features
        q = self.encoder_q(x1)  # queries: B,C,1,1,1
        q = nn.functional.normalize(q, dim=1)
        q = q.view(B, self.dim)

        in_train_mode = q.requires_grad

        # compute key features
        with torch.no_grad():  # no gradient to keys
            if in_train_mode: self._momentum_update_key_encoder()  # update the key encoder

            # # shuffle for making use of BN
            # x2, idx_unshuffle = self._batch_shuffle_ddp(x2)

            k = self.encoder_k(x2)  # keys: B,C,1,1,1
            k = nn.functional.normalize(k, dim=1)

            # # undo shuffle
            # k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        k = k.view(B, self.dim)

        # compute logits
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: B,(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        
        # dequeue and enqueue
        if in_train_mode: self._dequeue_and_enqueue(k)

        return logits, labels






class UberNCE(InfoNCE):
    '''
    UberNCE is a supervised version of InfoNCE,
    it uses labels to define positive and negative pair
    Still, use MoCo to enlarge the negative pool
    '''
    def __init__(self, network='s3d', dim=128, K=2048, m=0.999, T=0.07):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 2048)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(UberNCE, self).__init__(network, dim, K, m, T)
        # extra queue to store label
        self.register_buffer("queue_label", torch.ones(K, dtype=torch.long) * -1)


    def forward(self, block, k_label):
        '''Output: logits, binary mask for positive pairs
        '''
        (B, N, *_) = block.shape # [B,N,C,T,H,W]
        assert N == 2
        x1 = block[:,0,:].contiguous()
        x2 = block[:,1,:].contiguous()

        # compute query features
        q = self.encoder_q(x1)  # queries: B,C,1,1,1
        q = nn.functional.normalize(q, dim=1)
        q = q.view(B, self.dim)

        in_train_mode = q.requires_grad

        # compute key features
        with torch.no_grad():  # no gradient to keys
            if in_train_mode: self._momentum_update_key_encoder()  # update the key encoder

            # # shuffle for making use of BN
            # x2, idx_unshuffle = self._batch_shuffle_ddp(x2)

            k = self.encoder_k(x2)  # keys: B,C,1,1,1
            k = nn.functional.normalize(k, dim=1)

            # # undo shuffle
            # k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        k = k.view(B, self.dim)

        # compute logits
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: B,(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # mask: binary mask for positive keys
        mask = k_label.unsqueeze(1) == self.queue_label.unsqueeze(0) # B,K
        mask = torch.cat([torch.ones((mask.shape[0],1), dtype=torch.long, device=mask.device).bool(),
                          mask], dim=1) # B,(1+K)
                
        # dequeue and enqueue
        if in_train_mode: self._dequeue_and_enqueue(k, k_label)

        return logits, mask

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, k_labels):
        # gather keys before updating queue
        # keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        self.queue_label[ptr:ptr+batch_size] = k_labels
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr




## pytorch default: torch.nn.BatchNorm3d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
## tensorflow s3d code: torch.nn.BatchNorm3d(num_features, eps=1e-3, momentum=0.001, affine=True, track_running_stats=True)

class BasicConv3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv3d, self).__init__()
        self.conv = nn.Conv3d(in_planes, out_planes, 
                              kernel_size=kernel_size, stride=stride, 
                              padding=padding, bias=False)

        # self.bn = nn.BatchNorm3d(out_planes, eps=1e-3, momentum=0.001, affine=True)
        self.bn = nn.BatchNorm3d(out_planes)
        self.relu = nn.ReLU(inplace=True)

        # init
        self.conv.weight.data.normal_(mean=0, std=0.01) # original s3d is truncated normal within 2 std
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class STConv3d(nn.Module):
    def __init__(self,in_planes,out_planes,kernel_size,stride,padding=0):
        super(STConv3d, self).__init__()
        if isinstance(stride, tuple):
            t_stride = stride[0]
            stride = stride[-1]
        else: # int
            t_stride = stride
            
        self.conv1 = nn.Conv3d(in_planes, out_planes, kernel_size=(1,kernel_size,kernel_size),
                              stride=(1,stride,stride),padding=(0,padding,padding), bias=False)
        self.conv2 = nn.Conv3d(out_planes,out_planes,kernel_size=(kernel_size,1,1),
                               stride=(t_stride,1,1),padding=(padding,0,0), bias=False)

        # self.bn1=nn.BatchNorm3d(out_planes, eps=1e-3, momentum=0.001, affine=True)
        # self.bn2=nn.BatchNorm3d(out_planes, eps=1e-3, momentum=0.001, affine=True)
        self.bn1=nn.BatchNorm3d(out_planes)
        self.bn2=nn.BatchNorm3d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        
        # init
        self.conv1.weight.data.normal_(mean=0, std=0.01) # original s3d is truncated normal within 2 std
        self.conv2.weight.data.normal_(mean=0, std=0.01) # original s3d is truncated normal within 2 std
        self.bn1.weight.data.fill_(1)
        self.bn1.bias.data.zero_()
        self.bn2.weight.data.fill_(1)
        self.bn2.bias.data.zero_()
    
    def forward(self,x):
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)
        x=self.conv2(x)
        x=self.bn2(x)
        x=self.relu(x)
        return x


class SelfGating(nn.Module):
    def __init__(self, input_dim):
        super(SelfGating, self).__init__()
        self.fc = nn.Linear(input_dim, input_dim)

    def forward(self, input_tensor):
        """Feature gating as used in S3D-G"""
        spatiotemporal_average = torch.mean(input_tensor, dim=[2, 3, 4])
        weights = self.fc(spatiotemporal_average)
        weights = torch.sigmoid(weights)
        return weights[:, :, None, None, None] * input_tensor


class SepInception(nn.Module):
    def __init__(self, in_planes, out_planes, gating=False):
        super(SepInception, self).__init__()

        assert len(out_planes) == 6
        assert isinstance(out_planes, list)

        [num_out_0_0a, 
        num_out_1_0a, num_out_1_0b,
        num_out_2_0a, num_out_2_0b, 
        num_out_3_0b] = out_planes

        self.branch0 = nn.Sequential(
            BasicConv3d(in_planes, num_out_0_0a, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(in_planes, num_out_1_0a, kernel_size=1, stride=1),
            STConv3d(num_out_1_0a, num_out_1_0b, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(in_planes, num_out_2_0a, kernel_size=1, stride=1),
            STConv3d(num_out_2_0a, num_out_2_0b, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(in_planes, num_out_3_0b, kernel_size=1, stride=1),
        )

        self.out_channels = sum([num_out_0_0a, num_out_1_0b, num_out_2_0b, num_out_3_0b])

        self.gating = gating 
        if gating:
            self.gating_b0 = SelfGating(num_out_0_0a)
            self.gating_b1 = SelfGating(num_out_1_0b)
            self.gating_b2 = SelfGating(num_out_2_0b)
            self.gating_b3 = SelfGating(num_out_3_0b)


    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        if self.gating:
            x0 = self.gating_b0(x0)
            x1 = self.gating_b1(x1)
            x2 = self.gating_b2(x2)
            x3 = self.gating_b3(x3)

        out = torch.cat((x0, x1, x2, x3), 1)

        return out


class S3D(nn.Module):

    def __init__(self, input_channel=3, gating=False, slow=False):
        super(S3D, self).__init__()
        self.gating = gating 
        self.slow = slow 

        if slow:
            self.Conv_1a = STConv3d(input_channel, 64, kernel_size=7, stride=(1,2,2), padding=3)
        else: # normal
            self.Conv_1a = STConv3d(input_channel, 64, kernel_size=7, stride=2, padding=3) 

        self.block1 = nn.Sequential(self.Conv_1a) # (64, 32, 112, 112)
            
        ###################################

        self.MaxPool_2a = nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)) 
        self.Conv_2b = BasicConv3d(64, 64, kernel_size=1, stride=1) 
        self.Conv_2c = STConv3d(64, 192, kernel_size=3, stride=1, padding=1) 

        self.block2 = nn.Sequential(
            self.MaxPool_2a, # (64, 32, 56, 56)
            self.Conv_2b,    # (64, 32, 56, 56)
            self.Conv_2c)    # (192, 32, 56, 56)

        ###################################
        
        self.MaxPool_3a = nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)) 
        self.Mixed_3b = SepInception(in_planes=192, out_planes=[64, 96, 128, 16, 32, 32], gating=gating)
        self.Mixed_3c = SepInception(in_planes=256, out_planes=[128, 128, 192, 32, 96, 64], gating=gating)

        self.block3 = nn.Sequential(
            self.MaxPool_3a,    # (192, 32, 28, 28)
            self.Mixed_3b,      # (256, 32, 28, 28)
            self.Mixed_3c)      # (480, 32, 28, 28)

        ###################################
        
        self.MaxPool_4a = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        self.Mixed_4b = SepInception(in_planes=480, out_planes=[192, 96, 208, 16, 48, 64], gating=gating)
        self.Mixed_4c = SepInception(in_planes=512, out_planes=[160, 112, 224, 24, 64, 64], gating=gating)
        self.Mixed_4d = SepInception(in_planes=512, out_planes=[128, 128, 256, 24, 64, 64], gating=gating)
        self.Mixed_4e = SepInception(in_planes=512, out_planes=[112, 144, 288, 32, 64, 64], gating=gating)
        self.Mixed_4f = SepInception(in_planes=528, out_planes=[256, 160, 320, 32, 128, 128], gating=gating)

        self.block4 = nn.Sequential(
            self.MaxPool_4a,  # (480, 16, 14, 14)
            self.Mixed_4b,    # (512, 16, 14, 14)
            self.Mixed_4c,    # (512, 16, 14, 14)
            self.Mixed_4d,    # (512, 16, 14, 14)
            self.Mixed_4e,    # (528, 16, 14, 14)
            self.Mixed_4f)    # (832, 16, 14, 14)

        ###################################
        
        self.MaxPool_5a = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0))
        self.Mixed_5b = SepInception(in_planes=832, out_planes=[256, 160, 320, 32, 128, 128], gating=gating)
        self.Mixed_5c = SepInception(in_planes=832, out_planes=[384, 192, 384, 48, 128, 128], gating=gating)

        self.block5 = nn.Sequential(
            self.MaxPool_5a,  # (832, 8, 7, 7)
            self.Mixed_5b,    # (832, 8, 7, 7)
            self.Mixed_5c)    # (1024, 8, 7, 7)

        ###################################

        # self.AvgPool_0a = nn.AvgPool3d(kernel_size=(2, 7, 7), stride=1)
        # self.Dropout_0b = nn.Dropout3d(dropout_keep_prob)
        # self.Conv_0c = nn.Conv3d(1024, num_classes, kernel_size=1, stride=1, bias=True)

        # self.classifier = nn.Sequential(
        #     self.AvgPool_0a,
        #     self.Dropout_0b,
        #     self.Conv_0c)
        

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x 

