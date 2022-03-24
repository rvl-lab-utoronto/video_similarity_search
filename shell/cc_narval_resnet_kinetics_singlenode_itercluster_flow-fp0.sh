#!/bin/bash
#SBATCH --account=def-florian7_gpu 
#SBATCH --time=3-23:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=kin_16fr_r3d
#SBATCH --output=%x-%j.out
#SBATCH --gres=gpu:4
#SBATCH --mem=498G
#SBATCH --cpus-per-task=48
# --wait-all-nodes=1


module load python/3.6
source /home/cheny257/projects/def-florian7/cheny257/code/venv/bin/activate


tar -xzf /home/cheny257/projects/def-florian7/datasets/kinetics400/frames_shortedge320px_25fps/val_split.tar.gz -C $SLURM_TMPDIR
echo 'Extracted val zip'
tar -xzf /home/cheny257/projects/def-florian7/datasets/kinetics400/frames_shortedge320px_25fps/train_split.tar.gz -C $SLURM_TMPDIR
echo 'Extracted train zip'

#flow
tar -xzf /home/cheny257/projects/def-florian7/datasets/kinetics400/FLOW/FLOW-u-jpg.tar.gz -C $SLURM_TMPDIR
echo 'Extracted flow train zip'

cd $SLURM_TMPDIR

#export MASTER_ADDRESS=$(hostname)
#echo $MASTER_ADDRESS

#MPORT=3456
#echo $MPORT

ROOTDIR=/home/cheny257/projects/def-florian7/cheny257/code/video_similarity_search 


python -u $ROOTDIR/online_train.py \
--iterative_cluster \
--cfg $ROOTDIR/config/custom_configs/resnet_kinetics_itercluster_flow_cc.yaml \
--gpu 0,1,2,3 --num_data_workers 4 \
--batch_size 140 \
--output $ROOTDIR/output_ucf29-kin-r3d-single \
--epoch 401 VAL.BATCH_SIZE 80 LOSS.LOCAL_LOCAL_CONTRAST True \
DATA.INPUT_CHANNEL_NUM 3 ITERCLUSTER.METHOD finch FINCH_PARTITION [0,3] \
DATA.SAMPLE_DURATION 16

#--checkpoint_path $ROOTDIR/output_ucf16-adam-32/tnet_checkpoints/3dresnet/checkpoint.pth.tar
