#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=p05-32f-s3d
#SBATCH --output=%x-%j.out
#SBATCH --partition=rtx6000
#SBATCH --qos=normal
#SBATCH --gres=gpu:4
#SBATCH --mem=167G
#SBATCH --cpus-per-task=40
# --wait-all-nodes=1
# --time=2-23:00:00

source ~/ENV1/bin/activate

export MASTER_ADDRESS=$(hostname)
echo $MASTER_ADDRESS
MPORT=3456
echo $MPORT

ROOTDIR=/h/salarh/RVL/repos/video_similarity_search

# create a symbolic link to link the checkpoint directory under your working dir
ln -sfn /checkpoint/${USER}/${SLURM_JOB_ID} $PWD/checkpoint

# ask the system to preserve the checkpoint directory for 48 hours after job done
touch /checkpoint/${USER}/${SLURM_JOB_ID}/DELAYPURGE

OUTDIR=$ROOTDIR/output_ucf30-kin-s3d-p05-32f-warmup

#touch $OUTDIR/log.out

#bs=10/gpu for s3d on 15G gpu for 32 frames
#bs=16/gpu for s3d on 24G gpu for 32 frames

python -u $ROOTDIR/online_train.py --vector --cfg $ROOTDIR/config/custom_configs/resnet_kin_itercluster_flow_vector.yaml --gpu 0,1,2,3 --num_data_workers 4 --batch_size 64 --output $OUTDIR --epoch 1 --ip_address_port tcp://$MASTER_ADDRESS:$MPORT --checkpoint_path $PWD/checkpoint VAL.BATCH_SIZE 40 DATA.SAMPLE_DURATION 32 DATASET.POSITIVE_SAMPLING_P 1.0 DATASET.SAMPLING_STRATEGY random_negative MODEL.ARCH s3d DATA.EVAL_BATCHSIZE_MULTIPLIER 9 DATASET.CLUSTER_PATH /scratch/ssd004/scratch/salarh/datasets/kinetics400/vid_clusters.txt
#--num_shards 2 --ip_address_port tcp://$MASTER_ADDRESS:$MPORT --compute_canada
#|& tee -a $OUTDIR/log.out
#--checkpoint_path $ROOTDIR/output_ucf16-adam-32/tnet_checkpoints/3dresnet/checkpoint.pth.tar
