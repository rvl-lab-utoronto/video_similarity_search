#!/bin/bash
#SBATCH --time=2-23:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=finch0
#SBATCH --output=%x-%j.out
#SBATCH --partition=t4v2
#SBATCH --qos=high
#SBATCH --gres=gpu:2
#SBATCH --mem=48G
#SBATCH --cpus-per-task=16
# --wait-all-nodes=1

#clush -w $(slurm_hl2hl.py --format PDSH) tar -xzf /home/salar77h/projects/def-florian7/datasets/UCF101/poolnet_new.tar.gz -C $SLURM_TMPDIR
#echo 'Extracted mask zip'

#clush -w $(slurm_hl2hl.py --format PDSH) tar -xzf /home/salar77h/projects/def-florian7/datasets/UCF101/ucf101_tvl1_flow-2.tar.gz -C $SLURM_TMPDIR
#echo 'Extracted ucf101_tvl1_flow-2 zip'
#clush -w $(slurm_hl2hl.py --format PDSH) tar -xzf /home/salar77h/projects/def-florian7/datasets/UCF101/jpg.tar.gz -C $SLURM_TMPDIR
#echo 'Extracted rgb zip'

#cd $SLURM_TMPDIR

#export MASTER_ADDRESS=$(hostname)
#echo $MASTER_ADDRESS

#MPORT=3456
#echo $MPORT

#ROOTDIR=/home/salar77h/projects/def-florian7/salar77h/repos/video_similarity_search 
ROOTDIR=/h/salarh/RVL/repos/video_similarity_search

# create a symbolic link to link the checkpoint directory under your working dir
ln -sfn /checkpoint/${USER}/${SLURM_JOB_ID} $PWD/checkpoint

# ask the system to preserve the checkpoint directory for 48 hours after job done
touch /checkpoint/${USER}/${SLURM_JOB_ID}/DELAYPURGE

python -u $ROOTDIR/online_train.py --vector --iterative_cluster --cfg $ROOTDIR/config/custom_configs/resnet_ucf_itercluster_flow_vector.yaml --gpu 0,1 --num_data_workers 4 --batch_size 32 --output $ROOTDIR/output_ucf23-iterclus --epoch 601 --checkpoint_path $PWD/checkpoint VAL.BATCH_SIZE 80
#--checkpoint_path $ROOTDIR/output_ucf16-adam-32/tnet_checkpoints/3dresnet/checkpoint.pth.tar
