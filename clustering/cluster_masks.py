# Example usage:
# python clustering/cluster_masks.py --cfg config/custom_configs/slowfast_kinetics.yaml \
# --gpu 2 --split train --output cluster_output_kin_train --num_data_workers 8 \
# DATA.SAMPLE_SIZE 224 DATA.SAMPLE_DURATION 16 TRAIN.BATCH_SIZE 128


import os
import argparse
import torch
import torchvision.models as models
import cv2
import numpy as np
from torchvision import transforms
from sklearn.cluster import AgglomerativeClustering, DBSCAN, OPTICS, KMeans
from sklearn.metrics import normalized_mutual_info_score
import time
import pickle as pkl

from config.m_parser import load_config, arg_parser
from datasets import data_loader
from datasets.spatial_transforms import (Compose, Resize, CenterCrop, ToTensor)
# from spherecluster import SphericalKMeans

#https://github.com/jasonlaska/spherecluster
#from spherecluster import SphericalKMeans

np.random.seed(1)

# Argument parser
def m_arg_parser(parser):
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Visualize salient masks only'
    )
    parser.add_argument(
        '--embedding_dir',
        type=str,
        default=None,
        help='Directory holding already-processed embeddings of salient-masked center frames'
    )
    parser.add_argument(
        '--label_dir',
        type=str,
        default=None,
        help='Directory holding true labels pkl'
    )
    parser.add_argument(
        '--split',
        type=str,
        default=None,
        help='dataset split to cluster'
    )
    parser.add_argument(
        '--method',
        type=str,
        default='kmeans',
        help='Clustering algorithm'
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


def get_embeddings_mask_regions(model, data, test_loader, log_interval=2, visualize=False):
    print("Getting embeddings...")

    model.eval()
    embeddings = []
    #vid_paths = []

    MASK_THRESHOLD = 0.01

    # Mean and std for ImageNet
    # https://pytorch.org/docs/stable/torchvision/models.html
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    with torch.no_grad():
        for batch_idx, (inputs, _, _) in enumerate(test_loader): 
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

                # Put image values into range [0, 1] and then normalize
                center_img_salient = tensor_min_max_normalize(center_img_salient)
                center_img_salient = normalize(center_img_salient)
                center_img_salient_batch.append(center_img_salient)
            center_img_salient = torch.stack(center_img_salient_batch, dim=0)  # N x 3 channels x 1 frame x H x W

            #print ('Input size', inputs.size())      
            #print ('salient', center_img_salient.size())    
            #print(vid_path)

            if (visualize):
                # Visualize mask region (use batch size 1)
                center_img_salient = center_img_salient.squeeze(0)
                center_img = vid_tensor_to_numpy(center_img_salient.unsqueeze(1))[0]
                center_img = cv2.cvtColor(center_img, cv2.COLOR_RGB2BGR)
                center_img = cv_f32_to_u8(center_img)
                cv2.imshow('input', center_img)
                cv2.waitKey()

                ## Visualize mask
                ##mask = vid_tensor_to_numpy(input[3].unsqueeze(0))[0]
                ##print(mask)
                ##mask = cv2.merge([mask, mask, mask])
            else:
                if cuda:
                    center_img_salient = center_img_salient.to(device)

                embedd = model(center_img_salient)
                embeddings.append(embedd.detach().cpu())
                #vid_paths.extend(vid_path)

                if (batch_idx == 0):
                    print('Embedd size', embedd.size())

                if (batch_idx+1) % log_interval == 0:
                    print('Encoded [{}/{}]'.format((batch_idx+1)*batch_size, len(data)))

    embeddings = torch.cat(embeddings, dim=0)
    embeddings = embeddings.squeeze()  # remove dimensions with size 1 (from avg pooling)
    return embeddings


# Preprocessing for kmeans - l2normalize
def preprocess_features_kmeans(data):
    l2norms = torch.norm(data, dim=1, keepdim=True)
    data = data / l2norms
    print('l2-normalized data')
    return data


# Perform clustering 
def fit_cluster(embeddings, method='Agglomerative', k=1000, l2normalize=True):


    assert(method in ['DBSCAN', 'Agglomerative', 'OPTICS', 'kmeans', 'spherical_kmeans'])

    print("Clustering with {}...".format(method))
    if method == 'kmeans':
        print("k:", k)

    if method == 'Agglomerative':
        distance_threshold = 0.24 #0.24 for ucf train
        trained_cluster_obj = AgglomerativeClustering(n_clusters=None,
                                                      linkage='average',
                                                      distance_threshold=distance_threshold,
                                                      affinity='cosine').fit(embeddings)
    elif method == 'DBSCAN':
        # If small clusters have too many incorrect increase min_samples, if there are some very large clusters with
        # too many incorrect decrease eps, if too few / little cluster representation decrease min_samples, if too many -1 increase eps
        trained_cluster_obj = DBSCAN(eps=0.14, # 0.18 for ucf val, #0.14 for ucf train, #0.12 for kin train
                                     min_samples=2, #3 for ucf val, #2 for ucf train, 3 for kin train
                                     metric='cosine',
                                     n_jobs=-1).fit(embeddings)
    elif method == 'kmeans':
        #pre-process - l2 normalize embeddings
        if l2normalize:
            embeddings = preprocess_features_kmeans(embeddings)

        n_clusters = k #2000 for ucf train
        trained_cluster_obj = KMeans(n_clusters=n_clusters,
                                     n_init=10).fit(embeddings)
    # elif method == 'spherical_kmeans':
    #     n_clusters = k
    #     print('clustering with spherical kmeans with k={}'.format(n_clusters))
    #     trained_cluster_obj = SphericalKMeans(n_clusters=n_clusters).fit(embeddings)
    elif method == 'OPTICS':
        trained_cluster_obj = OPTICS(min_samples=3, max_eps=0.20, cluster_method='dbscan', metric='cosine', n_jobs=-1).fit(embeddings)

    elif method == 'sphere':
        trained_cluster_obj = SphericalKMeans(n_clusters=k)
        trained_cluster_obj.fit(embeddings)

    labels = trained_cluster_obj.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    print("Fitted " + str(n_clusters) + " clusters with " + str(method))
    return trained_cluster_obj


# Print clusters 
def cluster_embeddings(clustering_obj, true_labels, label_to_class_map):

    labels = clustering_obj.labels_
    cluster_to_data_idxs = {label: np.where(clustering_obj.labels_ == label)[0] for label in set(labels)}

    # Put each data with cluster label -1 into its own new cluster
    if -1 in cluster_to_data_idxs:
        next_cluster_label = len(set(labels)) - 1
        for data_idx in cluster_to_data_idxs[-1]:
            cluster_to_data_idxs[next_cluster_label] = [data_idx]
            labels[data_idx] = next_cluster_label
            next_cluster_label += 1
        del cluster_to_data_idxs[-1]

    for cluster in cluster_to_data_idxs:
        cur_cluster_vids = []
        for data_idx in cluster_to_data_idxs[cluster]:
            vid_label = label_to_class_map[true_labels[data_idx]]
            cur_cluster_vids.append(vid_label)
        print(cluster, ':', cur_cluster_vids)

    return labels


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

    print("Starting...")

    # ======================= Imagenet-pretrained Model ========================

    img_model = models.resnet18(pretrained=True)

    # Discard FC (last kept is avg pool)
    img_model = torch.nn.Sequential(*(list(img_model.children())[:-1]))
    #print (img_model)

    if cuda:
        img_model.to(device)

    print('Finished generating resnet18 pretrained model')

    # ============================== Data Loaders ==============================

    if args.split is None:
        split = 'train'
    else:
        split = args.split

    spatial_transform = [
        Resize(cfg.DATA.SAMPLE_SIZE),
        CenterCrop(cfg.DATA.SAMPLE_SIZE),
        ToTensor()
    ]
    spatial_transform = Compose(spatial_transform)

    test_loader, (data, _) = data_loader.build_data_loader(split, cfg,
            triplets=False, req_spatial_transform=spatial_transform,
            req_train_shuffle=False, drop_last=False)
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
        embeddings = get_embeddings_mask_regions(img_model, data, test_loader, visualize=args.visualize)
        print('Time to get embeddings: {:.2f}s'.format(time.time()-start_time))

        with open(embedding_pkl_file, 'wb') as handle:
            torch.save(embeddings, handle, pickle_protocol=pkl.HIGHEST_PROTOCOL)
        print ('Embeddings saved to', embedding_pkl_file)

    print('Embeddings size', embeddings.size())
    print()

    # =============================== Clustering ===============================

    start_time = time.time()
    trained_clustering_obj = fit_cluster(embeddings, args.method)
    print('Time to cluster: {:.2f}s'.format(time.time()-start_time))

    if args.label_dir:
        labels_pkl = os.path.join(args.label_dir, 'labels.pkl')
        with open(labels_pkl, 'rb') as handle:
            true_labels = torch.load(handle)
        print ('Labels loaded from', labels_pkl)
    else:
        true_labels = data.get_total_labels()

    NMI = normalized_mutual_info_score(true_labels, trained_clustering_obj.labels_)

    cluster_labels = cluster_embeddings(trained_clustering_obj, true_labels,
            data.get_label_to_class_map())

    print('NMI between true labels and cluster assignments: {:.2f}\n'.format(NMI))

    with open(os.path.join(args.output, 'vid_clusters.txt'), "a") as f:
        for label in cluster_labels:
            f.write('{}\n'.format(label))
        print('Saved cluster labels to', os.path.join(args.output, 'vid_clusters.txt'))

