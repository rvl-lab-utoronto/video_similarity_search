#!/bin/bash
#SBATCH --account=def-florian7_gpu 
#SBATCH --time=0-17:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:t4:1
#SBATCH --mem=32G
#SBATCH --job-name=slowfast_kinetics
#SBATCH --output=%x-%j.out

cd $SLURM_TMPDIR
mkdir work_kinetics
cd work_kinetics
tar -xzf /home/salar77h/projects/def-florian7/datasets/kinetics400/frames_shortedge320px_25fps/val_split.tar.gz
echo 'Extracted val zip'
tar -xzf /home/salar77h/projects/def-florian7/datasets/kinetics400/frames_shortedge320px_25fps/train_split.tar.gz
echo 'Extracted train zip'

python /home/salar77h/projects/def-florian7/salar77h/repos/video_similarity_search/train.py --cfg /home/salar77h/projects/def-florian7/salar77h/repos/video_similarity_search/config/custom_configs/slowfast_kinetics_cc.yaml
