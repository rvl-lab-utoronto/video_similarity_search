import torch
import cv2
import numpy as np
from torch.autograd import Function
import torch.nn.functional as F

from models.model_utils import model_selector, multipathway_input
from config.m_parser import load_config, arg_parser


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
            print(name)
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
        if self.cuda:
            self.model = model.cuda()

        self.extractor1 = ModelOutputs(self.model, self.feature_module, target_layer_names, self.model_arch)
        self.extractor2 = ModelOutputs(self.model, self.feature_module, target_layer_names, self.model_arch)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input1, input2):
        if self.cuda:
            if self.model_arch == 'slowfast':
                for i in range(len(input1)):
                    input1[i], input2[i] = input1[i].to(device), input2[i].to(device)
                
                for i in range(len(input1)):
                    print('input size:', input1[i].size())
            else:
                print('input size:', input1.size())

        features1, output1 = self.extractor1(input1)
        print()
        features2, output2 = self.extractor2(input2)

        for i in range(len(features1)):
            print('features',features1[i].size())
        print('output', output1.size())

        similarity = 1.0 / F.pairwise_distance(output1, output2, 2)
        self.feature_module.zero_grad()
        self.model.zero_grad()
        similarity.backward(retain_graph=True)

        grads_val1 = self.extractor1.get_gradients()[-1].cpu().data.numpy()
        grads_val2 = self.extractor2.get_gradients()[-1].cpu().data.numpy()
        print('gradval shape:',grads_val1.shape)
        #print(grads_val1[0][1])
        #print(grads_val1[0][2])
        #print()
        #print(grads_val2[0][1])
        #print(grads_val2[0][2])

        target1 = features1[-1]
        target1 = target1.cpu().data.numpy()[0, :]
        target2 = features2[-1]
        target2 = target2.cpu().data.numpy()[0, :]
        print('target shape:', target1.shape)

        '''


        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        print('weights shape:', weights.shape)
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        print('cam shape:', cam.shape)

        # only take positive - only interested in features with positive
        # influence on class of interest, i.e. pixels whose intensity should be
        # increased in order to increased probability of this class
        cam = np.maximum(cam, 0)

        # upsample - bilinear interpolation by default
        cam = cv2.resize(cam, input.shape[2:])

        # Min-Max feature scaling - bring all values into range [0,1]
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)

        print('upsampled cam shape:',cam.shape)
        return cam
        '''
        return None


if __name__ == '__main__':
    args = arg_parser().parse_args()
    cfg = load_config(args)

    cuda = torch.cuda.is_available()
    global device; device = torch.cuda.current_device()

    model = model_selector(cfg)
    #print(model._modules.items())

    grad_cam = VideoSimilarityGradCam(model=model, feature_module=model.s5_fuse, \
                            target_layer_names=[], use_cuda=cuda, model_arch=cfg.MODEL.ARCH)

    x = torch.randn(1, 3, cfg.DATA.SAMPLE_DURATION, cfg.DATA.SAMPLE_SIZE, \
                    cfg.DATA.SAMPLE_SIZE, dtype=torch.float, requires_grad=True)
    y = torch.randn(1, 3, cfg.DATA.SAMPLE_DURATION, cfg.DATA.SAMPLE_SIZE, \
                    cfg.DATA.SAMPLE_SIZE, dtype=torch.float, requires_grad=True)
    if cfg.MODEL.ARCH == 'slowfast':
        x = multipathway_input(x, cfg)
        y = multipathway_input(y, cfg)
    
    print()
    mask = grad_cam(x, y)
    





