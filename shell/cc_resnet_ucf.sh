#!/bin/bash
#SBATCH --account=def-florian7_gpu
#SBATCH --time=0-18:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --job-name=resnet_ucf
#SBATCH --output=%x-%j.out

cd $SLURM_TMPDIR
mkdir work_ucf
cd work_ucf
tar -xzf /home/cheny257/projects/def-florian7/datasets/UCF101/jpg.tar.gz
echo 'Extracted jpg.tar.gz'

python /home/cheny257/projects/def-florian7/cheny257/code/video_similarity_search/train.py --cfg /home/cheny257/projects/def-florian7/cheny257/code/video_similarity_search/config/custom_configs/resnet_ucf.yaml --output '/home/cheny257/projects/def-florian7/cheny257/output/ucf'
