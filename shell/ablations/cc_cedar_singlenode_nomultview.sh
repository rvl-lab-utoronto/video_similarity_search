#!/bin/bash
#SBATCH --account=def-florian7_gpu 
#SBATCH --time=0-23:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=abl-nomult
#SBATCH --output=%x-%j.out
#SBATCH --gres=gpu:v100l:2
#SBATCH --mem=48G
#SBATCH --cpus-per-task=16
# --wait-all-nodes=1

#clush -w $(slurm_hl2hl.py --format PDSH) tar -xzf /home/salar77h/projects/def-florian7/datasets/UCF101/poolnet_new.tar.gz -C $SLURM_TMPDIR
#echo 'Extracted mask zip'
clush -w $(slurm_hl2hl.py --format PDSH) tar -xzf /home/salar77h/projects/def-florian7/datasets/UCF101/ucf101_tvl1_flow-2.tar.gz -C $SLURM_TMPDIR
echo 'Extracted ucf101_tvl1_flow-2 zip'
clush -w $(slurm_hl2hl.py --format PDSH) tar -xzf /home/salar77h/projects/def-florian7/datasets/UCF101/jpg.tar.gz -C $SLURM_TMPDIR
echo 'Extracted rgb zip'

cd $SLURM_TMPDIR

#export MASTER_ADDRESS=$(hostname)
#echo $MASTER_ADDRESS

#MPORT=3456
#echo $MPORT

ROOTDIR=/home/salar77h/projects/def-florian7/salar77h/repos/video_similarity_search 

python $ROOTDIR/online_train.py --iterative_cluster --cfg $ROOTDIR/config/custom_configs/resnet_ucf_itercluster_flow_cc.yaml --gpu 0,1 --num_data_workers 4 --batch_size 32 --output $ROOTDIR/output_ucf23-nomultview --epoch 601 --checkpoint_path $ROOTDIR/output_ucf23-nomultview/tnet_checkpoints/3dresnet/checkpoint.pth.tar VAL.BATCH_SIZE 80 DATASET.POS_CHANNEL_REPLACE False DATASET.CHANNEL_EXTENSIONS none
#--checkpoint_path $ROOTDIR/output_ucf16-adam-32/tnet_checkpoints/3dresnet/checkpoint.pth.tar
