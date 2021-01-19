"""
Pulled from https://github.com/kenshohara/3D-ResNets-PyTorch
"""
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)

class ResizeConv3d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x

class BasicBlockEnc(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class BasicBlockDec(nn.Module):
    expansion = 1

    def __init__(self, in_planes, stride=1, downsample=None):
        super().__init__()
        self.planes = int(in_planes/stride)

        self.conv2 = conv3x3x3(in_planes, in_planes)
        self.bn2 = nn.BatchNorm3d(in_planes)

        if stride == 1:
            self.conv1 = nn.Conv3d(in_planes, self.planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm3d(self.planes)
            self.shortcut = nn.Sequential()
        else:
            self.conv1 = ResizeConv3d(in_planes, self.planes, kernel_size=3, scale_factor=stride)
            self.bn1 = nn.BatchNorm3d(self.planes)
            self.shortcut = nn.Sequential(
                ResizeConv3d(in_planes, self.planes, kernel_size=3, scale_factor=stride),
                nn.BatchNorm3d(self.planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn2(self.conv2(x)))
        out = self.bn1(self.conv1(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class ResNetEnc(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=3,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 hidden_layer= 2048,
                 out_dim = 128,
                 projection_head=True):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0] #64
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.projection_head = projection_head
        if projection_head:
            print('==> setting up non-linear project heads')
            self.fc1 = nn.Linear(block_inplanes[3] * block.expansion, hidden_layer)
            self.fc2 = nn.Linear(hidden_layer, out_dim)
        else:
            self.fc = nn.Linear(block_inplanes[3] * block.expansion, hidden_layer)


        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        print('\n\n==> encoder\n')
        print('x', x.shape)
        x = self.conv1(x)
        print('after conv1', x.shape)
        x = self.bn1(x)
        print('after bn1', x.shape)
        x = self.relu(x)
        print('after relu', x.shape)
        if not self.no_max_pool:
            x = self.maxpool(x)
        print('after maxpool', x.shape)

        x = self.layer1(x)
        print('after layer1', x.shape)
        x = self.layer2(x)
        print('after layer2', x.shape)
        x = self.layer3(x)
        print('after layer3', x.shape)
        x = self.layer4(x)
        print('after layer4', x.shape)

        x = self.avgpool(x)
        print('after avgpool', x.shape)
        x = x.view(x.size(0), -1)
        # x = self.fc(x)
        #add projection head
        print('after resize', x.shape)
        if self.projection_head:
            x = self.fc1(x)
            # print('after fc1', x.size())
            x = self.relu(x)
            # print('after relu', x.size())
            x = self.fc2(x)
            # print('after fc2', x.size())
        print('after projection heads', x.shape)
        return x

class ResNetDec(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=3,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 hidden_layer= 2048,
                 out_dim = 128,
                 projection_head=True):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[-1] #512
        self.no_max_pool = no_max_pool

        # if projection_head:
        #     print('==> setting up non-linear project heads')
        #     self.fc2 = nn.Linear(out_dim, hidden_layer)
        #     self.fc1 = nn.Linear(hidden_layer, block_inplanes[3] * block.expansion)
        # else:
        #     self.fc = nn.Linear(hidden_layer, block_inplanes[3] * block.expansion)

        self.linear = nn.Linear(out_dim, block_inplanes[-1])

        self.layer4 = self._make_layer(BasicBlockDec, 
                                       block_inplanes[2], 
                                       layers[3],
                                       shortcut_type,
                                       stride=2)

        self.layer3 = self._make_layer(BasicBlockDec, 
                                       block_inplanes[1],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer2 = self._make_layer(BasicBlockDec,
                                       block_inplanes[0],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)

        self.layer1 = self._make_layer(BasicBlockDec,
                                        block_inplanes[0],
                                        layers[0],
                                        shortcut_type,
                                        stride=1)
        self.conv1 = ResizeConv3d(64, 3, kernel_size=3, scale_factor=2)

        

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        strides = [stride, 1]# + [1]*(blocks-1) #why?        
        layers = []
        for stride in reversed(strides):
            layers += [block(in_planes=self.in_planes,
                  stride=stride,)]
        self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, z):
        print('\n\n==> Decoder \n')
        print(z.shape)
        x = self.linear(z)
        print('after linear', x.shape)
        x = x.view(z.size(0), 512, 1, 1, 1)
        print(x.shape)
        x = F.interpolate(x, size= (1,4,4))#scale_factor=4)
        print('after interpolate', x.shape)
        x = self.layer4(x)
        print('after layer4', x.shape)
        x = self.layer3(x)
        print('after layer3', x.shape)
        x = self.layer2(x)
        print('after layer2', x.shape)
        x = self.layer1(x)
        print('after layer1', x.shape)
        x = torch.sigmoid(self.conv1(x))
        print(x.shape)
        # x = x.view(x.size(0), 3, 64, 64) #(B, C, D, H, W)
        return x


class ResnetVAE(nn.Module):
    def __init__(self,  
                 n_input_channels=3,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 hidden_layer= 2048,
                 out_dim = 128,
                 projection_head=True):
        
        super().__init__()
        self.encoder = ResNetEnc(BasicBlockEnc, [2,2,2,2], self.get_inplanes(),
                                n_input_channels=3,
                                conv1_t_size=7,
                                conv1_t_stride=1,
                                no_max_pool=False,
                                shortcut_type='B',
                                widen_factor=1.0,
                                hidden_layer= 2048,
                                out_dim = 128,
                                projection_head=True)
        self.decoder = ResNetDec(BasicBlockDec, [2,2,2,2], self.get_inplanes(),
                                n_input_channels=3,
                                conv1_t_size=7,
                                conv1_t_stride=1,
                                no_max_pool=False,
                                shortcut_type='B',
                                widen_factor=1.0,
                                hidden_layer= 2048,
                                out_dim = 128,
                                projection_head=True)

    def get_inplanes(self):
        return [64, 128, 256, 512]

    

def generate_model(model_depth, **kwargs):
    def get_inplanes():
        return [64, 128, 256, 512]

    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = ResNet(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
    elif model_depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
    elif model_depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
    else:
        print('Not yet supported...')
    return model




if __name__ == '__main__':

    model_depth=200
    n_classes=1039
    n_input_channels=3
    resnet_shortcut = 'B'
    conv1_t_size = 7 #kernel size in t dim of conv1
    conv1_t_stride = 1 #stride in t dim of conv1
    no_max_pool = False #max pooling after conv1 is removed
    resnet_widen_factor = 1 #number of feature maps of resnet is multiplied by this value


    # model=generate_model(model_depth=model_depth, n_classes=n_classes,
    #                     n_input_channels=n_input_channels, shortcut_type=resnet_shortcut,
    #                     conv1_t_size=conv1_t_size,
    #                     conv1_t_stride=conv1_t_stride,
    #                     no_max_pool=no_max_pool,
    #                     widen_factor=resnet_widen_factor)

    # print(model)
    x = torch.rand((1,3,16,112,112))
    model=ResnetVAE(n_input_channels=n_input_channels, shortcut_type=resnet_shortcut,
                        conv1_t_size=conv1_t_size,
                        conv1_t_stride=conv1_t_stride,
                        no_max_pool=no_max_pool,
                        widen_factor=resnet_widen_factor)
    print(model.encoder)
    print(model.decoder)

    output = model.encoder(x)
    z = model.decoder(output)
    print(output.shape)
    print(z.shape)