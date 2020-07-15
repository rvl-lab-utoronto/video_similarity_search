"""
Created by Sherry Chen on Jul 14, 2020
retrieve the most similar clips
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import data_loader
from models.triplet_net import Tripletnet
from models.model_utils import model_selector, multipathway_input
from config.m_parser import load_config, parse_args

log_interval = 5
def evaluate(model, test_loader):
    model.eval()
    embedding = []
    with torch.no_grad():
        for batch_idx, (input, targets) in enumerate(test_loader):
            # print(input.size())
            batch_size = input.size(0)
            if cfg.MODEL.ARCH == 'slowfast':
                input = multipathway_input(input, cfg)
                if cuda:
                    for i in range(len(input)):
                        input= input[i].to(device)
            else:
                if cuda:
                    input= input.to(device)
            embedd = model(input)
            # print(embedd.size())
            embedding.append(embedd)
            if batch_idx % log_interval == 0:
                print('val [{}/{}]'.format(batch_idx * batch_size, len(test_loader.dataset)))
                
    embeddings = torch.cat(embedding, dim=0)

if __name__ == '__main__':
    args = parse_args()
    cfg = load_config(args)

    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    global cuda; cuda = torch.cuda.is_available()
    global device; device = torch.device("cuda" if cuda else "cpu")

    model=model_selector(cfg)
    print('=> finished generating {} backbone model...'.format(cfg.MODEL.ARCH))

    # Load pretrained backbone if path exists
    if args.pretrain_path is not None:
        model = load_pretrained_model(model, args.pretrain_path)

    if cuda:
        if torch.cuda.device_count() > 1:
            print("Let's use {} GPUs".format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)
    print('=> finished generating similarity network...')

    test_loader = data_loader.build_data_loader('val', cfg, triplets=False)
    evaluate(model, test_loader)
