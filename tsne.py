import os
import torch
import numpy as np
from sklearn.manifold import TSNE

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.decomposition import PCA

from datasets import data_loader
from config.m_parser import load_config, arg_parser
from models.model_utils import (model_selector, multipathway_input,
                            load_pretrained_model, save_checkpoint, load_checkpoint,
                            AverageMeter, accuracy, create_output_dirs)
from evaluate import evaluate
import pickle as pkl
from models.s3d.select_backbone import select_backbone
from coclr_utils.utils import neq_load_customized
from coclr_utils.classifier import LinearClassifier

if __name__ == '__main__':
    args = arg_parser().parse_args()
    cfg = load_config(args)

    print('GPU:', args.gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    # device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    device = torch.cuda.current_device()
    print('device:', device)
    cuda=True

    ###############

    train_loader, (train_data, _) = data_loader.build_data_loader('train', cfg,
            triplets=False, req_train_shuffle=False, drop_last=False)
    val_loader, (val_data, _) = data_loader.build_data_loader('val', cfg,
            triplets=False, val_sample=None, req_train_shuffle=False,
            drop_last=False)

    label_to_class_map = train_data.get_label_to_class_map()
    #print(label_to_class_map)

    NUM_CLASSES = 101
    split = 'train'

    ################
    feature_dir = 'tsne_feature_dir'

    embeddings_pkl = os.path.join(feature_dir, '{}_embeddings.pkl'.format(split))
    idxs_pkl = os.path.join(feature_dir, '{}_idxs.pkl'.format(split))
    labels_pkl = os.path.join(feature_dir, '{}_labels.pkl'.format(split))

    COMPUTE_EMBEDDINGS = False

    if COMPUTE_EMBEDDINGS:

        if cfg.MODEL.ARCH == 'coclr':
            print('getting coclr embeddings')
            #model, param = select_backbone('s3d')
            #feature_size = param['feature_size']
            model = LinearClassifier(
                    network='s3d', 
                    num_class=101,
                    dropout=0.9,
                    use_dropout=False,
                    use_final_bn=True,
                    use_l2_norm=True)

            model = model.cuda(device=device)

            model = torch.nn.DataParallel(model)
            model_without_dp = model.module

            checkpoint = torch.load(args.checkpoint_path, map_location=torch.device('cpu'))
            epoch = checkpoint['epoch']
            state_dict = checkpoint['state_dict']

            #if args.retrieval: # if directly test on pretrained network
            new_dict = {}
            for k,v in state_dict.items():
                k = k.replace('encoder_q.0.', 'backbone.')
                new_dict[k] = v
            state_dict = new_dict

            try: model_without_dp.load_state_dict(state_dict)
            except: neq_load_customized(model_without_dp, state_dict, verbose=True)

            #print(model_without_dp)

        else:
            model=model_selector(cfg, projection_head=False, classifier=False)
            #print(model)
            #exit()
            start_epoch, best_acc = load_checkpoint(model, args.checkpoint_path, classifier=True)
            model = model.cuda(device=device)
            print("start_epoch:{}, best_acc:{}".format(start_epoch, best_acc))


        if split == 'train':
            print('getting train embeddings')
            embeddings, labels, idxs = evaluate(cfg, model, cuda, device,
                    train_loader, split='train', is_master_proc=True)
        elif split == 'val':
            print('getting val embeddings')
            embeddings, labels, idxs = evaluate(cfg, model, cuda, device,
                    val_loader, split='val', is_master_proc=True)


        with open(embeddings_pkl, 'wb') as handle:
            torch.save(embeddings, handle, pickle_protocol=pkl.HIGHEST_PROTOCOL)
        with open(labels_pkl, 'wb') as handle:
            torch.save(labels, handle, pickle_protocol=pkl.HIGHEST_PROTOCOL)
        with open(idxs_pkl, 'wb') as handle:
            torch.save(idxs, handle, pickle_protocol=pkl.HIGHEST_PROTOCOL)
        print('saved {}_embeddings'.format(split), embeddings.size(), 'labels', len(labels))

        exit()



    if os.path.exists(embeddings_pkl) and os.path.exists(labels_pkl) and os.path.exists(idxs_pkl):
        with open(embeddings_pkl, 'rb') as handle:
            embeddings = torch.load(handle)
        with open(labels_pkl, 'rb') as handle:
            labels = torch.load(handle)
        with open(idxs_pkl, 'rb') as handle:
            idxs = torch.load(handle)

    labels = np.asarray(labels)

    print(embeddings.shape)
    print(labels.shape)

    print(len(set(labels)))

    X = embeddings
    y = labels

    labels = [label_to_class_map[i] for i in y]

    feat_cols = [ 'pixel'+str(i) for i in range(X.shape[1]) ]
    df = pd.DataFrame(X,columns=feat_cols)
    df['y'] = y
    #df['label'] = df['y'].apply(lambda i: str(i))
    #df['label'] = df['y'].apply(lambda i: label_to_class_map[i])
    df['label'] = labels
    X, y = None, None
    print('Size of the dataframe: {}'.format(df.shape))

    # For reproducability of the results
    np.random.seed(42)
    #rndperm = np.random.permutation(df.shape[0])

    #N = 10000
    #df_subset = df.loc[rndperm[:N],:].copy()

    rndperm = np.random.permutation(NUM_CLASSES)
    N_CLASS_SUBSET = 20
    class_subset = rndperm[:N_CLASS_SUBSET]

    #df = df.sample(n=10,axis='y')

    df_subset = df.loc[df['y'].isin(class_subset)].copy()
    #df_subset = df
    data_subset = df_subset[feat_cols].values

    print(df_subset.shape)

    ######### PCA ########

    pca_50 = PCA(n_components=50)
    pca_result_50 = pca_50.fit_transform(data_subset)
    print('Cumulative explained variation for 50 principal components:{}'.format(np.sum(pca_50.explained_variance_ratio_)))


    #####################

    time_start = time.time()
    #tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    #tsne_results = tsne.fit_transform(data_subset)
    tsne_results = TSNE(n_components=2, perplexity=30).fit_transform(pca_result_50)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

    df_subset['tsne-2d-one'] = tsne_results[:,0]
    df_subset['tsne-2d-two'] = tsne_results[:,1]
    plt.figure(figsize=(16,10))
    #plt.figure()
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="label",
        palette=sns.color_palette("hls", N_CLASS_SUBSET),
        data=df_subset,
        legend="full",
        alpha=0.3
    )

    plt.savefig('tsne_{}.png'.format(split), dpi=300)
    #plt.show()

    #print(X_embedded.shape)
