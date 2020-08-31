import os
import argparse
import torch
import torchvision.models as models
import cv2
import numpy as np
from torchvision import transforms
from sklearn.cluster import AgglomerativeClustering
import time
import pickle as pkl

from config.m_parser import load_config, arg_parser
from datasets import data_loader
from datasets.temporal_transforms import TemporalCenterFrame, TemporalSpecificCrop, TemporalEndFrame
from datasets.spatial_transforms import (Compose, Resize, CenterCrop, ToTensor)


np.random.seed(1)

# Argument parser
def m_arg_parser(parser):
    parser.add_argument(
        '--name',
        type=str,
        default=None,
        help='Please specify the name (e.g. ResNet18_K, SlowFast_U): '
    )
    parser.add_argument(
        '--embedding_dir',
        type=str,
        default=None,
        help='Directory holding already-processed embeddings of salient-masked center frames'
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


def get_embeddings_mask_regions(model, data, test_loader, log_interval=5):
    print("Getting embeddings...")
    
    model.eval()
    embeddings = []
    vid_paths = []

    #center_temporal_transform = TemporalCenterFrame()
    #first_temporal_transform = TemporalSpecificCrop(0, size=1)
    #last_temporal_transform = TemporalEndFrame()
    #last_temporal_transform = TemporalSpecificCrop(0, size=1)

    MASK_THRESHOLD = 0.01

    with torch.no_grad():      
        for batch_idx, (inputs, _, vid_path) in enumerate(test_loader): 
            num_frames = inputs.shape[2]
            rgb_center_img_tensor = inputs[:, :3, num_frames // 2, :, :]  # N x 3 x H x W
            masks = inputs[:,3,:,:,:]  
            mask = torch.mean(masks, dim=1)  # N x H x W
            #mask[i] = torch.ceil(mask[i])
            #mask[i] = torch.round(mask[i])

            batch_size = inputs.size(0)

            center_img_salient_batch = []
            for i in range(batch_size):
                # If little to none of image is salient, use entire image
                mask_mean = torch.mean(mask[i])
                if mask_mean < MASK_THRESHOLD:
                    center_img_salient = rgb_center_img_tensor[i]
                else:
                    center_img_salient = rgb_center_img_tensor[i] * mask[i]

                # Put image values into range [0, 1] and then normalize using 
                # mean and std for ImageNet
                # https://pytorch.org/docs/stable/torchvision/models.html
                center_img_salient = tensor_min_max_normalize(center_img_salient)
                normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
                center_img_salient = normalize(center_img_salient)
                center_img_salient_batch.append(center_img_salient)
            center_img_salient = torch.stack(center_img_salient_batch, dim=0)  # N x 3 channels x 1 frame x H x W

            #print ('Input size', inputs.size())      
            #print ('salient', center_img_salient.size())    
            #print(vid_path)

            # Visualize mask region (use batch size 1)
            #center_img_salient = center_img_salient.squeeze(0)
            #center_img = vid_tensor_to_numpy(center_img_salient.unsqueeze(1))[0]
            #center_img = cv2.cvtColor(center_img, cv2.COLOR_RGB2BGR)
            #center_img = cv_f32_to_u8(center_img)
            #cv2.imshow('input', center_img)
            #cv2.waitKey()

            ## Visualize mask
            ##mask = vid_tensor_to_numpy(input[3].unsqueeze(0))[0]
            ##print(mask)
            ##mask = cv2.merge([mask, mask, mask])

            if cuda:
                center_img_salient = center_img_salient.to(device)

            embedd = model(center_img_salient)
            embeddings.append(embedd.detach().cpu())
            vid_paths.extend(vid_path)

            if (batch_idx == 0):
                print('Embedd size', embedd.size())

            if (batch_idx+1) % log_interval == 0:
                print('Encoded [{}/{}]'.format((batch_idx+1)*batch_size, len(data)))

    embeddings = torch.cat(embeddings, dim=0)
    embeddings = embeddings.squeeze()  # remove dimensions with size 1 (from avg pooling)
    return embeddings, vid_paths


def fit_cluster(embeddings, method='AgglomerativeClustering', distance_threshold=0.95):
    print("Clustering...")

    distance_threshold = 0.24

    if method == 'AgglomerativeClustering':
        trained_cluster_obj = AgglomerativeClustering(n_clusters=None,
                                                      linkage='average',
                                                      distance_threshold=distance_threshold,
                                                      affinity='cosine').fit(embeddings)
        labels = trained_cluster_obj.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    print("Fitted " + str(n_clusters) + " clusters with " + str(method))
    return trained_cluster_obj


def cluster_embeddings(data, clustering_obj):
    
    clusters = {}
    
    for idx, (_, _, vid_path) in enumerate(data): 
        #print (vid_path, 'cluster:', clustering_obj.labels_[idx])
        cluster_label = clustering_obj.labels_[idx]
        if cluster_label not in clusters:
            clusters[cluster_label] = []

        vid_label = vid_path.split(os.sep)[-2]
        clusters[cluster_label].append(vid_label)

    for idx, cluster in enumerate(clusters):
        print(idx, ':', clusters[cluster])


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

    if not os.path.exists(cfg.OUTPUT_PATH):
        os.makedirs(cfg.OUTPUT_PATH)

    # ======================= Imagenet-pretrained Model ========================

    img_model = models.resnet18(pretrained=True)

    # Discard FC (last kept is avg pool)
    img_model = torch.nn.Sequential(*(list(img_model.children())[:-1]))
    #print (img_model)

    if cuda:
        img_model.to(device)

    print('Finished generating resnet18 pretrained model')

    # ============================== Data Loaders ==============================

    split = 'val'

    spatial_transform = [
        Resize(cfg.DATA.SAMPLE_SIZE),
        CenterCrop(cfg.DATA.SAMPLE_SIZE),
        ToTensor()
    ]
    spatial_transform = Compose(spatial_transform)

    test_loader, data = data_loader.build_data_loader(split, cfg, triplets=False, req_spatial_transform=spatial_transform, req_train_shuffle=False)
    print()

    # =============================== Embeddings ===============================

    if args.embedding_dir:
        embedding_pkl_file = os.path.join(args.embedding_dir, 'embedding_file.pkl')
    else:
        embedding_pkl_file = os.path.join(args.output, 'embedding_file.pkl')

    if args.embedding_dir and os.path.isfile(embedding_pkl_file):
        with open(embedding_pkl_file, 'rb') as handle:
            embeddings = torch.load(handle)
        print ('Embeddings loaded from', embedding_pkl_file)
    else:
        start_time = time.time()
        embeddings, _ = get_embeddings_mask_regions(img_model, data, test_loader)
        print('Time to get embeddings: {:.2f}s'.format(time.time()-start_time))
        
        with open(embedding_pkl_file, 'wb') as handle:
            torch.save(embeddings, handle, pickle_protocol=pkl.HIGHEST_PROTOCOL)
        print ('Embeddings saved to', embedding_pkl_file)

    print('Embeddings size', embeddings.size())
    print()
    
    # =============================== Clustering ===============================

    start_time = time.time()
    trained_clustering_obj = fit_cluster(embeddings)
    print('Time to cluster: {:.2f}s'.format(time.time()-start_time))

    cluster_embeddings(data, trained_clustering_obj)

