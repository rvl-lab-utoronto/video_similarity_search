#!/bin/bash
#SBATCH --account=def-florian7_gpu
#SBATCH --time=3-20:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=finch_resnet_ucf_ic
#SBATCH --output=%x-%j.out
#SBATCH --gres=gpu:v100l:2
#SBATCH --mem=40G
#SBATCH --cpus-per-task=16
# --wait-all-nodes=1

module load python/3.6

source /home/cheny257/projects/def-florian7/cheny257/code/resnet_env/bin/activate

echo 'extracting jpg.tar.gz'
clush -w $(slurm_hl2hl.py --format PDSH) tar -xf /home/cheny257/projects/def-florian7/datasets/UCF101/jpg.tar.gz -C $SLURM_TMPDIR
echo 'Done'



cd $SLURM_TMPDIR


# export MASTER_ADDRESS=$(hostname)
# echo master_address:$MASTER_ADDRESS

# MPORT=3457
# echo master_port:$MPORT

srun python /home/cheny257/projects/def-florian7/cheny257/code/video_similarity_search/online_train.py \
--cfg '/home/cheny257/projects/def-florian7/cheny257/code/video_similarity_search/config/custom_configs/ablation/finch/cc_resnet_ucf_ic.yaml' \
--gpu 0,1 \
--num_data_workers 4 \
--batch_size 32 \
--output '/home/cheny257/projects/def-florian7/cheny257/output/ablation/finch/resnet_ucf_ic' \
# --num_shards 2 \
--epoch 601 \
# --ip_address_port tcp://$MASTER_ADDRESS:$MPORT \
--compute_canada \
--iterative_cluster \
