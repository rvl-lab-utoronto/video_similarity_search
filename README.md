# video_similarity_search

## Dataset
* UCF101
  * RGB:  [LMDB](http://thor.robots.ox.ac.uk/~vgg/data/CoCLR/ucf101_rgb_lmdb.tar)
  * TVL1 Optical Flow: [LMDB](http://thor.robots.ox.ac.uk/~vgg/data/CoCLR/ucf101_flow_lmdb.tar)
* Kinetics

## Pretrain
* ResNet-18 pretrain on UCF-RGB & Optical Flow w/ Iterative Clustering and Temporal Discrimination Loss
 
```
python online_train.py --cfg config/custom_configs/resnet_ucf_itercluster_flow.yaml \
--gpu 0,1 --batch_size 32 --output ./output/path --iterative_cluster
```

## Finetune (Adapted from CoCLR)

* end-to-end finetune
```
python coclr_classify.py --cfg config/custom_configs/resnet_ucf_itercluster_flow.yaml \
--train_what ft --epochs 150 --schedule 60 100 --pretrain/resume /pth/to/ckpt.pth.tar --gpu 0 
```
* linear probe
```
python coclr_classify.py --cfg config/custom_configs/resnet_ucf_itercluster_flow.yaml \
--train_what last --epochs 150 --schedule 60 100 --pretrain/resume /pth/to/ckpt.pth.tar --gpu 0
```
* test
```
python coclr_classify.py --cfg config/custom_configs/resnet_ucf_itercluster_flow.yaml \
--train_what ft --checkpoint_path /pth/to/ckpt.pth.tar --ten_crop --gpu 0
```

## Evaluate retrieval (Adapted from IIC)

```
python iic_retrieve_clips.py --checkpoint_path /pth/to/ckpt.pth.tar --cfg config/custom_configs/resnet_ucf_itercluster_flow.yaml --gpu 0 --feature_dir eval_output --dataset ucf101
```


