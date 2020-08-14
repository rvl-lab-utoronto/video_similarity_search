#!/bin/bash
#SBATCH --account=def-florian7_gpu
#SBATCH --time=0-20:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=resnet_K_evaluate
#SBATCH --output=%x-%j.out
#SBATCH --gres=gpu:v100l:4
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
# --wait-all-nodes=1

module load python/3.6

source /home/cheny257/projects/def-florian7/cheny257/code/resnet_env/bin/activate


clush -w $(slurm_hl2hl.py --format PDSH) tar -xvf /home/cheny257/projects/def-florian7/datasets/UCF101/jpg.tar.gz -C $SLURM_TMPDIR
echo 'Extracted jpg.tar.gz'
cd $SLURM_TMPDIR


python /home/cheny257/projects/def-florian7/cheny257/code/video_similarity_search/evaluate.py \
--name ResNet18_K_MULTINODE_08112020 \
--num_exemplar 10 \
--cfg '/home/cheny257/projects/def-florian7/cheny257/code/video_similarity_search/config/custom_configs/cc_resnet_ucf.yaml' \
--gpu 0,1,2,3 \
--num_data_workers 4 \
--batch_size 40 \
--checkpoint_path '/home/cheny257/projects/def-florian7/cheny257/output/ResNet18_K_multinode_20hr/tnet_checkpoints/3dresnet/checkpoint.pth.tar' \
--output '/home/cheny257/projects/def-florian7/cheny257/output/ResNet18_K_multinode_20hr' \
