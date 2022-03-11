#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --job-name=p05-32f-s3d-multi
#SBATCH --output=%x-%j.out
#SBATCH --partition=rtx6000
#SBATCH --qos=normal
#SBATCH --gres=gpu:4
#SBATCH --mem=167G
#SBATCH --mincpus=40
# --cpus-per-task=40
# --wait-all-nodes=1
# --time=2-23:00:00

source ~/ENV1/bin/activate

# these commands don't need to run for all workers, put them here
MAIN_HOST=`hostname -s`
# this is the current host
export MASTER_ADDRESS=$MAIN_HOST
# pick a random available port
export MPORT="$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1])')"

# This is needed to avoid NCCL to use ifiniband, which the cluster does not have
export NCCL_IB_DISABLE=1
# This is to tell NCCL to use bond interface for network communication
if [[ "${SLURM_JOB_PARTITION}" == "t4v2" ]] || \
    [[ "${SLURM_JOB_PARTITION}" == "rtx6000" ]]; then
    echo export NCCL_SOCKET_IFNAME=bond0 on ${SLURM_JOB_PARTITION}
    export NCCL_SOCKET_IFNAME=bond0
fi

#export MASTER_ADDRESS=$(hostname)
#echo $MASTER_ADDRESS
#MPORT=3456
#echo $MPORT

ROOTDIR=/h/salarh/RVL/repos/video_similarity_search

# create a symbolic link to link the checkpoint directory under your working dir
ln -sfn /checkpoint/${USER}/${SLURM_JOB_ID} $PWD/checkpoint32-multi

# ask the system to preserve the checkpoint directory for 48 hours after job done
touch /checkpoint/${USER}/${SLURM_JOB_ID}/DELAYPURGE

OUTDIR=$ROOTDIR/output_ucf30-kin-s3d-p05-32f-warmup-2nodes

#touch $OUTDIR/log.out

#bs=10/gpu for s3d on 15G gpu for 32 frames
#bs=16/gpu for s3d on 24G gpu for 32 frames

srun --mem=167G python -u $ROOTDIR/online_train.py --vector --iterative_cluster --cfg $ROOTDIR/config/custom_configs/resnet_kin_itercluster_flow_vector.yaml --gpu 0,1,2,3 --num_data_workers 4 --batch_size 64 --output $OUTDIR --epoch 301 --num_shards 2 --ip_address_port tcp://$MASTER_ADDRESS:$MPORT --checkpoint_path $PWD/checkpoint32-multi --vector_init_checkpoint $OUTDIR/tnet_checkpoints/s3d/checkpoint.pth.tar --compute_canada VAL.BATCH_SIZE 40 DATA.SAMPLE_DURATION 32 DATASET.POSITIVE_SAMPLING_P 0.5 MODEL.ARCH s3d DATA.EVAL_BATCHSIZE_MULTIPLIER 9 TRAIN.CHECKPOINT_FREQ 0.03
#--num_shards 2 --ip_address_port tcp://$MASTER_ADDRESS:$MPORT --compute_canada
#|& tee -a $OUTDIR/log.out
#--checkpoint_path $ROOTDIR/output_ucf16-adam-32/tnet_checkpoints/3dresnet/checkpoint.pth.tar
