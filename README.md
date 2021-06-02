# video_similarity_search


## Pretrain
* ResNet-18 pretrain on UCF-RGB & Optical Flow w/ Iterative Clustering and Temporal Discrimination Loss
* 
```
python online_train.py --cfg config/custom_configs/resnet_ucf_itercluster_optical_llc.yaml --gpu 0,1 --batch_size 32 --output ~/output/resnet_ucf_itercluster_optical_llc --iterative_cluster
```

## Finetune
### Method 1 (Adapted from IIC/VCOP/CMC)
* train
```
python ft_classify.py -cfg config/custom_configs/resnet_ucf.yaml --checkpoint_path /pth/to/ckpt.pth.tar --gpu 1
```
* test
```
python ft_classify.py -cfg config/custom_configs/resnet_ucf.yaml --checkpoint_path /pth/to/ckpt.pth.tar --gpu 1 --mode test
```
* plot confusion matrix (after running test)
```
python ft_classify.py -cfg config/custom_configs/resnet_ucf.yaml --checkpoint_path /pth/to/ckpt.pth.tar --gpu 1 --mode plot
```
### Method 2 (Adapted from CoCLR)

*train
```
CUDA_VISIBLE_DEVICES=1,2 python coclr_classify.py --cfg config/custom_configs/resnet_ucf.yaml --train_what ft --epochs 200 --schedule 60 100 --pretrain/resume /pth/to/ckpt.pth.tar 
```
*test
```
CUDA_VISIBLE_DEVICES=1,2 python coclr_classify.py --cfg config/custom_configs/resnet_ucf.yaml --train_what ft --checkpoint_path /pth/to/ckpt.pth.tar --ten_crop 
```



