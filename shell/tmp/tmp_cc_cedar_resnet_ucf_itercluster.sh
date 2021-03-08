#!/bin/bash
#SBATCH --account=def-florian7_gpu
#SBATCH --time=0-20:10:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=triplet_ucf_itercluster_f_scratch_lr0.1
#SBATCH --output=%x-%j.out
#SBATCH --gres=gpu:v100l:4
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
# --wait-all-nodes=1

module load python/3.6

source /home/cheny257/projects/def-florian7/cheny257/vidsim_env/bin/activate


clush -w $(slurm_hl2hl.py --format PDSH) tar -xvf /home/cheny257/projects/def-florian7/datasets/UCF101/jpg.tar.gz -C $SLURM_TMPDIR
echo 'Extracted jpg.tar.gz'


cd $SLURM_TMPDIR


export MASTER_ADDRESS=$(hostname)
echo master_address:$MASTER_ADDRESS

MPORT=3456
echo master_port:$MPORT

srun python /home/cheny257/projects/def-florian7/cheny257/code/video_similarity_search/online_train.py \
--cfg '/home/cheny257/projects/def-florian7/cheny257/code/video_similarity_search/config/custom_configs/cc_resnet_ucf_itercluster.yaml' \
--gpu 0,1,2,3 \
--num_data_workers 4 \
--batch_size 120 \
--output '/home/cheny257/projects/def-florian7/cheny257/output/triplet_itercluster_lr0.1' \
--checkpoint '/home/cheny257/projects/def-florian7/cheny257/output/triplet_itercluster_lr0.1/tnet_checkpoints/3dresnet/checkpoint.pth.tar' \
--num_shards 2 \
--epoch 800 \
--ip_address_port tcp://$MASTER_ADDRESS:$MPORT \
--compute_canada \
--iterative_cluster
