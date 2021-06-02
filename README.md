# video_similarity_search


##Pretrain
* ResNet-18 pretrain on UCF-RGB & Optical Flow w/ Iterative Clustering and Temporal Discrimination Loss

'''
python online_train.py --cfg config/custom_configs/resnet_ucf_itercluster_optical_llc.yaml --gpu 0,1 --batch_size 32 --output ~/output/resnet_ucf_itercluster_optical_llc --iterative_cluster
'''
