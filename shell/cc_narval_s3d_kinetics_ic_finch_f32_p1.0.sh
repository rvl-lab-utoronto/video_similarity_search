#!/bin/bash
#SBATCH --account=def-florian7_gpu 
#SBATCH --time=1-23:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=kin_f32_s3d_p1.0
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
--cfg $ROOTDIR/config/custom_configs/s3d_kinetics.yaml \
--gpu 0,1,2,3 --num_data_workers 4 \
--batch_size 152 \
--output $ROOTDIR/output-kin-s3d-single_f32_1.0 \
--epoch 2 VAL.BATCH_SIZE 80 LOSS.LOCAL_LOCAL_CONTRAST True \
DATA.INPUT_CHANNEL_NUM 3 ITERCLUSTER.METHOD finch