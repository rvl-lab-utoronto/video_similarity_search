
# video_similarity_search

## Dataset
For pre-training, we follow the instruction on [this repo](https://github.com/kenshohara/3D-ResNets-PyTorch) to install and pre-process UCF101, HMDB51, and Kinetics400.
For evaluation, we download the LMDB version for UCF101 from the following links:
* UCF101
  * RGB:  [LMDB](http://thor.robots.ox.ac.uk/~vgg/data/CoCLR/ucf101_rgb_lmdb.tar)
  * TVL1 Optical Flow: [LMDB](http://thor.robots.ox.ac.uk/~vgg/data/CoCLR/ucf101_flow_lmdb.tar)

## Pretrain
* ResNet-18 pretrain on UCF-RGB & Optical Flow w/ Iterative Clustering and Temporal Discrimination Loss
 
```
python online_train.py --cfg config/custom_configs/resnet_ucf_itercluster_optical_llc.yaml \
--gpu 0,1 --batch_size 32 --output ~/output/path --iterative_cluster
```
* generate train/val acc & cluster quality plot (NMI)

```
python misc/generate_report.py -p
```

* Nearest Neighbor Retrieval
```
python evaluate.py --cfg config/custom_configs/resnet_ucf_itercluster_optical_llc.yaml \
--checkpoint_path /pth/to/ckpt.pth.tar --gpu 0 
```

## Finetune
### Method 1 (Adapted from IIC/VCOP/CMC)
* end-to-end finetune
```
python ft_classify.py -cfg config/custom_configs/resnet_ucf.yaml --checkpoint_path /pth/to/ckpt.pth.tar \
--gpu 1
```

* test
```
python ft_classify.py -cfg config/custom_configs/resnet_ucf.yaml --checkpoint_path /pth/to/ckpt.pth.tar \
--gpu 1 --mode test
```
* plot confusion matrix (after running test)
```
python ft_classify.py -cfg config/custom_configs/resnet_ucf.yaml --checkpoint_path /pth/to/ckpt.pth.tar \
--gpu 1 --mode plot
```
### Method 2 (Adapted from CoCLR)

* end-to-end finetune
```
python coclr_classify.py --cfg config/custom_configs/resnet_ucf.yaml --train_what ft \
--epochs 200 --schedule 60 100 --pretrain/resume /pth/to/ckpt.pth.tar --gpu 0 
```
* linear probe
```
python coclr_classify.py -cfg config/custom_configs/resnet_ucf.yaml --train_what last \
--epochs 200 --schedule 60 100 --pretrain/resume /pth/to/ckpt.pth.tar --gpu 0
```
* test
```
python coclr_classify.py --cfg config/custom_configs/resnet_ucf.yaml --train_what ft \
--checkpoint_path /pth/to/ckpt.pth.tar --ten_crop --gpu 0
```



