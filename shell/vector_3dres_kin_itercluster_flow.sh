#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=p07-32f
#SBATCH --output=%x-%j.out
#SBATCH --partition=rtx6000
#SBATCH --qos=normal
#SBATCH --gres=gpu:4
#SBATCH --mem=167G
#SBATCH --cpus-per-task=40
# --wait-all-nodes=1
# --time=2-23:00:00

ROOTDIR=/h/salarh/RVL/repos/video_similarity_search

# create a symbolic link to link the checkpoint directory under your working dir
ln -sfn /checkpoint/${USER}/${SLURM_JOB_ID} $PWD/checkpoint

# ask the system to preserve the checkpoint directory for 48 hours after job done
touch /checkpoint/${USER}/${SLURM_JOB_ID}/DELAYPURGE

python -u $ROOTDIR/online_train.py --vector --iterative_cluster --cfg $ROOTDIR/config/custom_configs/resnet_kin_itercluster_flow_vector.yaml --gpu 0,1,2,3 --num_data_workers 4 --batch_size 40 --output $ROOTDIR/output_ucf29-kin-r3d-p07-32f --epoch 401 --checkpoint_path $PWD/checkpoint VAL.BATCH_SIZE 80 DATA.SAMPLE_DURATION 32 DATASET.POSITIVE_SAMPLING_P 0.7
#--checkpoint_path $ROOTDIR/output_ucf16-adam-32/tnet_checkpoints/3dresnet/checkpoint.pth.tar
