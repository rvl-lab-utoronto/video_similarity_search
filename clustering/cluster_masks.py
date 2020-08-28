import os
import argparse
import torch
import torchvision.models as models
import cv2
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt

from config.m_parser import load_config, arg_parser
from datasets import data_loader
from datasets.temporal_transforms import TemporalCenterFrame, TemporalSpecificCrop, TemporalEndFrame

np.random.seed(1)

# Argument parser
def m_arg_parser(parser):
    parser.add_argument(
        '--name',
        type=str,
        default=None,
        help='Please specify the name (e.g. ResNet18_K, SlowFast_U): '
    )
    return parser


def vid_tensor_to_numpy(vid, is_batch=False):
    # channels x frames x width, height --> frames x width x height x channels
    vid_np = vid
    if is_batch:
        vid_np = vid[0]
    vid_np = vid_np.permute(1,2,3,0).numpy()
    return vid_np


def tensor_min_max_normalize(img):
    img = img - torch.min(img)
    img = img / torch.max(img)
    return img


def cv_f32_to_u8 (img):
    img = img - np.min(img)
    img = img / np.max(img)
    img = np.uint8(255 * img)
    return img


def get_embeddings_mask_regions(model, data, log_interval=5):
    model.eval()
    embeddings = []
    vid_paths = []

    #center_temporal_transform = TemporalCenterFrame()
    #first_temporal_transform = TemporalSpecificCrop(0, size=1)
    #last_temporal_transform = TemporalEndFrame()
    #last_temporal_transform = TemporalSpecificCrop(0, size=1)

    with torch.no_grad():
        for i in range (len(data)):
            # Get center frame
            #input, target, vid_path = data._get_video_custom_temporal(i, center_temporal_transform)
            #rgb_center_img_tensor = input[0:3]
            #mask = input[3]

            # Get full video
            #input, target, vid_path = data._get_video_custom_temporal(i)

            # Get beginning, middle, and end
            #input_center, target, vid_path = data._get_video_custom_temporal(i, center_temporal_transform)
            #input_first, target, vid_path = data._get_video_custom_temporal(i, first_temporal_transform)
            #input_last, target, vid_path = data._get_video_custom_temporal(i, last_temporal_transform)
            #rgb_center_img_tensor = input_center[0:3]
            #mask = (input_center[3] + input_first[3] + input_last[3]) / 2
            
            # Get cropped video
            input, target, vid_path = data.__getitem__(i)   # 4 channels x num_frames x H x W
            num_frames = input.shape[1]
            rgb_center_img_tensor = input[:3, num_frames // 2, :, :].unsqueeze(1)  # 3 x 1 x H x W
            masks = input[3]  
            mask = torch.mean(masks, dim=0)  # H x W
            #mask = torch.ceil(masks)
            #mask = torch.round(masks)

            center_img_salient = rgb_center_img_tensor*mask     # 3 channels x 1 frame x H x W
            center_img_salient = center_img_salient.squeeze(1)

            # Put image values into range [0, 1] and then normalize using 
            # mean and std for ImageNet
            # https://pytorch.org/docs/stable/torchvision/models.html

            center_img_salient = tensor_min_max_normalize(center_img_salient)
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
            center_img_salient = normalize(center_img_salient)

            #print ('Input size', input.size())      
            #print ('salient', center_img_salient.size())    
            #print(vid_path)

            # Visualize mask region
            #center_img = vid_tensor_to_numpy(center_img_salient.unsqueeze(1))[0]
            #center_img = cv2.cvtColor(center_img, cv2.COLOR_RGB2BGR)
            #center_img = cv_f32_to_u8(center_img)
            #cv2.imshow('input', center_img)
            #cv2.waitKey()

            ## Visual mask
            ##mask = vid_tensor_to_numpy(input[3].unsqueeze(0))[0]
            ##print(mask)
            ##mask = cv2.merge([mask, mask, mask])

            center_img_salient = center_img_salient.unsqueeze(0)
            if cuda:
                center_img_salient = center_img_salient.to(device)

            embedd = model(center_img_salient)
            embeddings.append(embedd.detach().cpu())
            vid_paths.extend(vid_path)

            if (i == 0):
                print('Embedd size', embedd.size())

            if i % log_interval == 0:
                print('Encoded [{}/{}]'.format(i, len(data)))

    embeddings = torch.cat(embeddings, dim=0)
    return embeddings, vid_paths


if __name__ == '__main__':
    args = m_arg_parser(arg_parser()).parse_args()
    cfg = load_config(args)

    force_data_parallel = True
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    global cuda; cuda = torch.cuda.is_available()
    global device; device = torch.device("cuda" if cuda else "cpu")

    if not args.output:
        output = input('Please specify output directory: ')
    else:
        output = args.output

    # ======================= Imagenet-pretrained Model ========================

    img_model = models.resnet18(pretrained=True)

    # Discard average pool and FC
    img_model = torch.nn.Sequential(*(list(img_model.children())[:-2]))
    #print (img_model)

    if cuda:
        img_model.to(device)

    print('Finished generating resnet18 pretrained model')

    # ============================== Data Loaders ==============================

    split = 'val'
    test_loader, data = data_loader.build_data_loader(split, cfg, triplets=False)
    print()

    # =============================== Embeddings ===============================

    embeddings, vid_info = get_embeddings_mask_regions(img_model, data)
