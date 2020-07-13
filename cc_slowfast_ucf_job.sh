#!/bin/bash
#SBATCH --account=def-florian7_gpu 
#SBATCH --time=0-17:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:t4:1
#SBATCH --mem=32G
#SBATCH --job-name=slowfast_ucf
#SBATCH --output=%x-%j.out

cd $SLURM_TMPDIR
mkdir work_ucf
cd work_ucf
tar -xzf /home/salar77h/projects/def-florian7/datasets/UCF101/jpg.tar.gz
echo 'Extracted jpg.tar.gz'

python /home/salar77h/projects/def-florian7/salar77h/repos/video_similarity_search/train.py --cfg /home/salar77h/projects/def-florian7/salar77h/repos/video_similarity_search/config/custom_configs/slowfast_ucf_cc.yaml
