#!/bin/bash
#SBATCH --account=def-florian7_gpu
#SBATCH --time=0-17:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:t4:2
#SBATCH --mem=32G
#SBATCH --job-name=resnet_kinetics
#SBATCH --output=%x-%j.out

cd $SLURM_TMPDIR
mkdir work_kinetics
cd work_kinetics
tar -xzf /home/cheny257/projects/def-florian7/datasets/kinetics400/frames_shortedge320px_25fps/val_split.tar.gz
echo 'Extracted val zip'
tar -xzf /home/cheny257/projects/def-florian7/datasets/kinetics400/frames_shortedge320px_25fps/train_split.tar.gz
echo 'Extracted train zip'
python /home/cheny257/projects/def-florian7/cheny257/code/video_similarity_search/train.py --cfg /home/cheny257/projects/def-florian7/cheny257/code/video_similarity_search/config/custom_configs/cc_resnet_kinetics.yaml --output '/home/cheny257/projects/def-florian7/cheny257/output/kinetics' --gpu 0,1
