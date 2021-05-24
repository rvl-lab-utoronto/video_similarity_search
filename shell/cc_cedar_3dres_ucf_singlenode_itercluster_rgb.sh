#!/bin/bash
#SBATCH --account=def-florian7_gpu 
#SBATCH --time=2-23:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=rgb2
#SBATCH --output=%x-%j.out
#SBATCH --gres=gpu:v100l:2
#SBATCH --mem=100G
#SBATCH --cpus-per-task=32
# --wait-all-nodes=1

#clush -w $(slurm_hl2hl.py --format PDSH) tar -xzf /home/salar77h/projects/def-florian7/datasets/UCF101/poolnet_new.tar.gz -C $SLURM_TMPDIR
#echo 'Extracted mask zip'
clush -w $(slurm_hl2hl.py --format PDSH) tar -xzf /home/salar77h/projects/def-florian7/datasets/UCF101/jpg.tar.gz -C $SLURM_TMPDIR
echo 'Extracted rgb zip'

cd $SLURM_TMPDIR

#export MASTER_ADDRESS=$(hostname)
#echo $MASTER_ADDRESS

#MPORT=3456
#echo $MPORT

ROOTDIR=/home/salar77h/projects/def-florian7/salar77h/repos/video_similarity_search 

#mkdir $ROOTDIR/output_ucf15-4gpu-s3d-4
#cp $ROOTDIR/output_ucf15-4gpu-s3d-3/vid_clusters.txt $ROOTDIR/output_ucf15-4gpu-s3d-4/

python $ROOTDIR/online_train.py --iterative_cluster --cfg $ROOTDIR/config/custom_configs/resnet_ucf_itercluster_cc.yaml --gpu 0,1 --num_data_workers 4 --batch_size 56 --output $ROOTDIR/output_ucf21-augfix-rgb-2gpu --epoch 601 VAL.BATCH_SIZE 80 LOSS.LOCAL_LOCAL_CONTRAST True
#--checkpoint_path $ROOTDIR/output_ucf16-adam-32/tnet_checkpoints/3dresnet/checkpoint.pth.tar
