#!/bin/bash
#SBATCH --account=def-florian7_gpu
#SBATCH --time=0-20:10:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=resnet_kinetics_multinodes
#SBATCH --output=%x-%j.out
#SBATCH --gres=gpu:v100l:4
#SBATCH --mem=48G
#SBATCH --cpus-per-task=10
# --wait-all-nodes=1


module load python/3.6

source /home/cheny257/projects/def-florian7/cheny257/code/resnet_env/bin/activate

clush -w $(slurm_hl2hl.py --format PDSH) tar -xzf /home/cheny257/projects/def-florian7/datasets/kinetics400/frames_shortedge320px_25fps/val_split.tar.gz -C $SLURM_TMPDIR
echo 'Extracted val zip'
clush -w $(slurm_hl2hl.py --format PDSH) tar -xzf /home/cheny257/projects/def-florian7/datasets/kinetics400/frames_shortedge320px_25fps/train_split.tar.gz -C $SLURM_TMPDIR
echo 'Extracted train zip'
cd $SLURM_TMPDIR


export MASTER_ADDRESS=$(hostname)
echo master_address:$MASTER_ADDRESS

MPORT=3457
echo master_port:$MPORT

srun python /home/cheny257/projects/def-florian7/cheny257/code/video_similarity_search/train.py \
--cfg '/home/cheny257/projects/def-florian7/cheny257/code/video_similarity_search/config/custom_configs/cc_resnet_kinetics.yaml' \
--gpu 0,1,2,3 \
--num_data_workers 4 \
--batch_size 40 \
--output '/home/cheny257/projects/def-florian7/cheny257/output/ResNet18_K_08192020' \
--checkpoint '/home/cheny257/projects/def-florian7/cheny257/output/ResNet18_K_08192020/tnet_checkpoints/3dresnet/checkpoint.pth.tar' \
--num_shards 2 \
--epoch 100 \
--ip_address_port tcp://$MASTER_ADDRESS:$MPORT \
--compute_canada
