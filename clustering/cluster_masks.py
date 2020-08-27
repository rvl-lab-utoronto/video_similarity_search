import os
import argparse
import torch
import torchvision.models as models
import cv2
import numpy as np

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


def cv_f32_to_u8 (img):
    img = img - np.min(img)
    img = img / np.max(img)
    img = np.uint8(255 * img)
    return img


def get_embeddings_mask_regions(model, data, log_interval=5):
    model.eval()
    embeddings = []
    vid_paths = []

    center_temporal_transform = TemporalCenterFrame()
    first_temporal_transform = TemporalSpecificCrop(0, size=1)
    last_temporal_transform = TemporalEndFrame()
    #last_temporal_transform = TemporalSpecificCrop(0, size=1)

    with torch.no_grad():
        for i in range (len(data)):
            # Get center frame
            '''input, target, vid_path = data._get_video_custom_temporal(i, center_temporal_transform)
            rgb_center_img_tensor = input[0:3]
            mask = input[3]'''

            # Get full video
            #input, target, vid_path = data._get_video_custom_temporal(i)
            # Get cropped video
            input, target, vid_path = data.__getitem__(i)  
            num_frames = input.shape[1]
            rgb_center_img_tensor = input[:3, num_frames // 2, :, :].unsqueeze(1)
            masks = input[3]
            #mask = torch.mean(masks, dim=0)
            mask = torch.mean(masks, dim=0)
            #mask = torch.ceil(masks)
            #mask = torch.round(masks)

            #mask = (masks[0] + masks[num_frames-1] + masks[num_frames // 2]) / 3

            # Get beginning, middle, and end
            #input_center, target, vid_path = data._get_video_custom_temporal(i, center_temporal_transform)
            #input_first, target, vid_path = data._get_video_custom_temporal(i, first_temporal_transform)
            #input_last, target, vid_path = data._get_video_custom_temporal(i, last_temporal_transform)
            #rgb_center_img_tensor = input_center[0:3]
            #mask = (input_center[3] + input_first[3] + input_last[3]) / 2

            print ('Input size', input.size())

            center_img = vid_tensor_to_numpy(rgb_center_img_tensor*mask)[0]
            center_img = cv2.cvtColor(center_img, cv2.COLOR_RGB2BGR)
            center_img = cv_f32_to_u8(center_img)

            #mask = vid_tensor_to_numpy(input[3].unsqueeze(0))[0]
            #print(mask)
            #mask = cv2.merge([mask, mask, mask])

            cv2.imshow('input', center_img)


            print(vid_path)
            cv2.waitKey()

            #break
         
            '''if cuda:
                input= input.to(device)

            embedd = model(input)
            embeddings.append(embedd.detach().cpu())
            vid_paths.extend(vid_path)
            print('Embedd size', embedd.size())

            if i % log_interval == 0:
                print('Encoded [{}/{}]'.format(i, len(data)))'''

    #embeddings = torch.cat(embeddings, dim=0)
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
    if cuda:
        img_model.to(device)

    print('Finished generating resnet18 pretrained model')

    # ============================== Data Loaders ==============================

    split = 'val'
    test_loader, data = data_loader.build_data_loader(split, cfg, triplets=False)
    print()

    # =============================== Embeddings ===============================

    embeddings, vid_info = get_embeddings_mask_regions(img_model, data)
