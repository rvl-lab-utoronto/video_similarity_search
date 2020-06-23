import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules.utils import _triple

# from module import SpatioTemporalConv


class Tripletnet(nn.Module):
    def __init__(self, embeddingnet):
        super(Tripletnet, self).__init__()
        self.embeddingnet = embeddingnet

    def forward(self, x, y, z):
        embedded_x = self.embeddingnet(x)
        embedded_y = self.embeddingnet(y)
        embedded_z = self.embeddingnet(z)
        dist_a = F.pairwise_distance(embedded_x, embedded_y, 2)
        dist_b = F.pairwise_distance(embedded_x, embedded_z, 2)
        return dist_a, dist_b, embedded_x, embedded_y, embedded_z

#
# class R2Plus1DNet(nn.Module):
#     r"""Forms the overall ResNet feature extractor by initializng 5 layers, with the number of blocks in
#     each layer set by layer_sizes, and by performing a global average pool at the end producing a
#     512-dimensional vector for each element in the batch.
#
#         Args:
#             layer_sizes (tuple): An iterable containing the number of blocks in each layer
#             block_type (Module, optional): Type of block that is to be used to form the layers. Default: SpatioTemporalResBlock.
#         """
#     def __init__(self, layer_sizes, block_type=SpatioTemporalResBlock):
#         super(R2Plus1DNet, self).__init__()
#
#         # first conv, with stride 1x2x2 and kernel size 3x7x7
#         self.conv1 = SpatioTemporalConv(3, 64, [3, 7, 7], stride=[1, 2, 2], padding=[1, 3, 3])
#         # output of conv2 is same size as of conv1, no downsampling needed. kernel_size 3x3x3
#         self.conv2 = SpatioTemporalResLayer(64, 64, 3, layer_sizes[0], block_type=block_type)
#         # each of the final three layers doubles num_channels, while performing downsampling
#         # inside the first block
#         self.conv3 = SpatioTemporalResLayer(64, 128, 3, layer_sizes[1], block_type=block_type, downsample=True)
#         self.conv4 = SpatioTemporalResLayer(128, 256, 3, layer_sizes[2], block_type=block_type, downsample=True)
#         self.conv5 = SpatioTemporalResLayer(256, 512, 3, layer_sizes[3], block_type=block_type, downsample=True)
#
#         # global average pooling of the output
#         self.pool = nn.AdaptiveAvgPool3d(1)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         x = self.conv4(x)
#         x = self.conv5(x)
#
#         x = self.pool(x)
#
#         return x.view(-1, 512)
#
# class R2Plus1DClassifier(nn.Module):
#     r"""Forms a complete ResNet classifier producing vectors of size num_classes, by initializng 5 layers,
#     with the number of blocks in each layer set by layer_sizes, and by performing a global average pool
#     at the end producing a 512-dimensional vector for each element in the batch,
#     and passing them through a Linear layer.
#
#         Args:
#             num_classes(int): Number of classes in the data
#             layer_sizes (tuple): An iterable containing the number of blocks in each layer
#             block_type (Module, optional): Type of block that is to be used to form the layers. Default: SpatioTemporalResBlock.
#         """
#     def __init__(self, num_classes, layer_sizes, block_type=SpatioTemporalResBlock):
#         super(R2Plus1DClassifier, self).__init__()
#
#         self.res2plus1d = R2Plus1DNet(layer_sizes, block_type)
#         self.linear = nn.Linear(512, num_classes)
#
#     def forward(self, x):
#         x = self.res2plus1d(x)
#         x = self.linear(x)
#
#         return x
#
#
#
#
# class SpatioTemporalConv(nn.Module):
#     r"""Applies a factored 3D convolution over an input signal composed of several input
#     planes with distinct spatial and time axes, by performing a 2D convolution over the
#     spatial axes to an intermediate subspace, followed by a 1D convolution over the time
#     axis to produce the final output.
#
#     Args:
#         in_channels (int): Number of channels in the input tensor
#         out_channels (int): Number of channels produced by the convolution
#         kernel_size (int or tuple): Size of the convolving kernel
#         stride (int or tuple, optional): Stride of the convolution. Default: 1
#         padding (int or tuple, optional): Zero-padding added to the sides of the input during their respective convolutions. Default: 0
#         bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
#     """
#
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
#         super(SpatioTemporalConv, self).__init__()
#
#         # if ints are entered, convert them to iterables, 1 -> [1, 1, 1]
#         kernel_size = _triple(kernel_size)
#         stride = _triple(stride)
#         padding = _triple(padding)
#
#         # decomposing the parameters into spatial and temporal components by
#         # masking out the values with the defaults on the axis that
#         # won't be convolved over. This is necessary to avoid unintentional
#         # behavior such as padding being added twice
#         spatial_kernel_size =  [1, kernel_size[1], kernel_size[2]]
#         spatial_stride =  [1, stride[1], stride[2]]
#         spatial_padding =  [0, padding[1], padding[2]]
#
#         temporal_kernel_size = [kernel_size[0], 1, 1]
#         temporal_stride =  [stride[0], 1, 1]
#         temporal_padding =  [padding[0], 0, 0]
#
#         # compute the number of intermediary channels (M) using formula
#         # from the paper section 3.5
#         intermed_channels = int(math.floor((kernel_size[0] * kernel_size[1] * kernel_size[2] * in_channels * out_channels)/ \
#                             (kernel_size[1]* kernel_size[2] * in_channels + kernel_size[0] * out_channels)))
#
#         # the spatial conv is effectively a 2D conv due to the
#         # spatial_kernel_size, followed by batch_norm and ReLU
#         self.spatial_conv = nn.Conv3d(in_channels, intermed_channels, spatial_kernel_size,
#                                     stride=spatial_stride, padding=spatial_padding, bias=bias)
#         self.bn = nn.BatchNorm3d(intermed_channels)
#         self.relu = nn.ReLU()
#
#         # the temporal conv is effectively a 1D conv, but has batch norm
#         # and ReLU added inside the model constructor, not here. This is an
#         # intentional design choice, to allow this module to externally act
#         # identical to a standard Conv3D, so it can be reused easily in any
#         # other codebase
#         self.temporal_conv = nn.Conv3d(intermed_channels, out_channels, temporal_kernel_size,
#                                     stride=temporal_stride, padding=temporal_padding, bias=bias)
#
#     def forward(self, x):
#         x = self.relu(self.bn(self.spatial_conv(x)))
#         x = self.temporal_conv(x)
#         return x
