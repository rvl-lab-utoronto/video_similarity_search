import os
import torch
import torch.nn as nn
import cv2
import numpy as np
from torch.autograd import Function
import torch.nn.functional as F
from scipy.ndimage import zoom

from models.triplet_net import Tripletnet
from models.model_utils import model_selector, multipathway_input
from config.m_parser import load_config, arg_parser
from datasets import data_loader
from train import load_checkpoint


class FeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers, model_arch):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []
        self.model_arch = model_arch

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []

        # Take single fused output path for slowfast
        if (self.model_arch == 'slowfast'):
            x = self.model(x)
            x = x[0]
            x.register_hook(self.save_gradient)
            outputs += [x]
            x = [x] # put in list to be compatible with future layers
        else:
            for name, module in self.model._modules.items():
                x = module(x)
                if name in self.target_layers:
                    x.register_hook(self.save_gradient)
                    outputs += [x]
        return outputs, x


class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, feature_module, target_layers, model_arch):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers, model_arch)
        self.model_arch = model_arch

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations = []
        for name, module in self.model._modules.items():
            #print(name)
            if self.model_arch == 'slowfast':
                if module == self.feature_module:
                    target_activations, x = self.feature_extractor(x)
                elif name == 'pathway0_pool':
                    x[0] = module(x[0])
                elif name == 'pathway1_pool':
                    x[1] = module(x[1])
                else:
                    x = module(x)
            else:
                print (self.model_arch, 'not yet supported')
                #if module == self.feature_module:
                #    target_activations, x = self.feature_extractor(x)
                #elif "avgpool" in name.lower():
                #    x = module(x)
                #    x = x.view(x.size(0),-1)
                #else:
                #    x = module(x)
                return
        
        return target_activations, x


class VideoSimilarityGradCam:
    def __init__(self, model, feature_module, target_layer_names, use_cuda, model_arch):
        self.model = model
        self.feature_module = feature_module
        self.model.eval()
        self.cuda = use_cuda
        self.model_arch = model_arch
        #if self.cuda:
        #    self.model = model.cuda()

        self.extractor1 = ModelOutputs(self.model, self.feature_module, target_layer_names, self.model_arch)
        self.extractor2 = ModelOutputs(self.model, self.feature_module, target_layer_names, self.model_arch)

    def forward(self, input):
        return self.model(input)

    def create_mask(self, features, grads_val, input_3d_shape):
        target = features[-1]
        target = target.cpu().data.numpy()[0, :]
        print('\ntarget shape:', target.shape)

        weights = np.mean(grads_val, axis=(2, 3, 4))[0, :]
        print('weights shape:', weights.shape)

        cam = np.zeros(target.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * target[i, :, :, :]
        print('cam shape:', cam.shape)

        # only take positive - only interested in features with positive
        # influence on class of interest, i.e. pixels whose intensity should be
        # increased in order to increased probability of this class
        cam = np.maximum(cam, 0)
        
        # spline interpolation
        scale = tuple(input_3d_shape[i] / cam.shape[i] for i in range(len(cam.shape)))
        cam = zoom(cam, scale)

        # trilinear interpolation
        #out_size = tuple(i for i in input_3d_shape)  # d*w*h
        #upsample = nn.Upsample(size=out_size, mode='trilinear')  
        #cam = torch.tensor(cam).unsqueeze(0).unsqueeze(0)  # n*c*d*w*h
        #cam = upsample(cam)
        #cam = cam[0][0].cpu().data.numpy()

        print('upsampled cam shape:', cam.shape)
        
        # Min-Max feature scaling - bring all values into range [0,1]
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)

        return cam

    def __call__(self, input1, input2):
        if self.cuda:
            if self.model_arch == 'slowfast':
                for i in range(len(input1)):
                    input1[i], input2[i] = input1[i].to(device), input2[i].to(device)
                
                for i in range(len(input1)):
                    print('input size:', input1[i].size())

                input_3d_shape = input1[1].shape[2:]
            else:
                input1, input2 = input1.to(device), input2.to(device)
                print('input size:', input1.size())
                input_3d_shape = input1.shape[2:]

        features1, output1 = self.extractor1(input1)
        features2, output2 = self.extractor2(input2)

        for i in range(len(features1)):
            print('features',features1[i].size())
        print('output', output1.size())

        similarity = 1.0 / F.pairwise_distance(output1, output2, 2)
        similarity = similarity.requires_grad_(True)

        self.feature_module.zero_grad()
        self.model.zero_grad()
        similarity.backward(retain_graph=True)

        grads_val1 = self.extractor1.get_gradients()[-1].cpu().data.numpy()
        grads_val2 = self.extractor2.get_gradients()[-1].cpu().data.numpy()
        print('gradval shape:', grads_val1.shape)

        cam1 = self.create_mask(features1, grads_val1, input_3d_shape)
        cam2 = self.create_mask(features2, grads_val2, input_3d_shape)

        return cam1, cam2


def get_img_and_cam(vid, masks, idx=None, alpha=0.4):
    if idx is None:
        idx = vid.shape[0] // 2  # take center image
    img = vid[idx]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    mask = masks[idx]

    # Mask must be type uint8 before applying color mask
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)

    # Need to convert from float32 to uint8 before using addWeighted, 
    # since heatmap type is uint8. Apply min max normalization since not certain
    # of the range of the values in the original image, and then put in range
    # 0 to 255.
    img = img - np.min(img)
    img = img / np.max(img)
    img = np.uint8(255 * img)

    cam = cv2.addWeighted(heatmap, alpha, img, 1.0-alpha, 0)

    return img, cam, heatmap


def show_cams_on_images(vid1, masks1, vid2, masks2):
    vid1 = vid1[0].permute(1,2,3,0).numpy()
    vid2 = vid2[0].permute(1,2,3,0).numpy()

    fps = 10.0
    while (True):
        for i in range(len(vid1)):
            img1_np, cam1, heatmap1 = get_img_and_cam(vid1, mask1, i)
            img2_np, cam2, heatmap2 = get_img_and_cam(vid2, mask2, i)

            imgs = np.vstack((img1_np, img2_np))
            cams = np.vstack((cam1, cam2))
            heatmaps = np.vstack((heatmap1, heatmap2))
            all_imgs = np.hstack((imgs, cams, heatmaps))

            cv2.imshow('Videos and their Similarity Heatmaps', all_imgs)
            cv2.waitKey(int(1.0/fps*1000.0))
        cv2.waitKey(2000)
    #cv2.waitKey()


# Argument parser
def m_arg_parser(parser):
    parser.add_argument(
        "--vid1",
        default=61,
        type=int,
        help='Video 1 dataset index'
    )
    parser.add_argument(
        "--vid2",
        default=62,
        type=int,
        help='Video 2 dataset index'
    )
    return parser


if __name__ == '__main__':
    args = m_arg_parser(arg_parser()).parse_args()
    cfg = load_config(args)

    np.random.seed(7)

    force_data_parallel = True
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    cuda = torch.cuda.is_available()
    global device; device = torch.device("cuda" if cuda else "cpu")

    # ============================== Model Setup ===============================

    model=model_selector(cfg)
    print('\n=> finished generating {} backbone model...'.format(cfg.MODEL.ARCH))

    tripletnet = Tripletnet(model)
    if cuda:
        cfg.NUM_GPUS = torch.cuda.device_count()
        print("Using {} GPU(s)".format(cfg.NUM_GPUS))
        if cfg.NUM_GPUS > 1 or force_data_parallel:
            tripletnet = nn.DataParallel(tripletnet)

    if args.checkpoint_path is not None:
        start_epoch, best_acc = load_checkpoint(tripletnet, args.checkpoint_path)

    model = tripletnet.module.embeddingnet
    #print(model._modules.items())
    if cuda:
        model.to(device)

    print('=> finished generating similarity network...')

    # ============================== Data Loaders ==============================

    test_loader, data = data_loader.build_data_loader('val', cfg, triplets=False)
    print()

    # ================================ Evaluate ================================

    grad_cam = VideoSimilarityGradCam(model=model, feature_module=model.s5_fuse, \
                            target_layer_names=[], use_cuda=cuda, model_arch=cfg.MODEL.ARCH)

    #x = torch.randn(1, 3, cfg.DATA.SAMPLE_DURATION, cfg.DATA.SAMPLE_SIZE, \
    #                cfg.DATA.SAMPLE_SIZE, dtype=torch.float, requires_grad=True)
    #y = torch.randn(1, 3, cfg.DATA.SAMPLE_DURATION, cfg.DATA.SAMPLE_SIZE, \
    #                cfg.DATA.SAMPLE_SIZE, dtype=torch.float, requires_grad=True)

    x_idx = args.vid1
    y_idx = args.vid2
    print('Img 1 path:', data.data[x_idx]['video'])
    print('Img 2 path:', data.data[y_idx]['video'])
    x, _, _ = data.__getitem__(x_idx)  # cropped size
    y, _, _ = data.__getitem__(y_idx)  # cropped size
    x = x.unsqueeze(0)
    y = y.unsqueeze(0)
    x_copy = x.clone().detach()
    y_copy = y.clone().detach()
    if cfg.MODEL.ARCH == 'slowfast':
        x = multipathway_input(x, cfg)
        y = multipathway_input(y, cfg)
    
    mask1, mask2 = grad_cam(x, y)
    show_cams_on_images(x_copy, mask1, y_copy, mask2)
    





