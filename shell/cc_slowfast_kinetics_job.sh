#!/bin/bash
#SBATCH --account=def-florian7_gpu 
#SBATCH --time=0-23:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:t4:4
#SBATCH --mem=48G
#SBATCH --job-name=slowfast_kinetics
#SBATCH --output=%x-%j.out

cd $SLURM_TMPDIR
mkdir work_kinetics
cd work_kinetics
tar -xzf /home/salar77h/projects/def-florian7/datasets/kinetics400/frames_shortedge320px_25fps/val_split.tar.gz
echo 'Extracted val zip'
tar -xzf /home/salar77h/projects/def-florian7/datasets/kinetics400/frames_shortedge320px_25fps/train_split.tar.gz
echo 'Extracted train zip'

python /home/salar77h/projects/def-florian7/salar77h/repos/video_similarity_search/train.py --cfg /home/salar77h/projects/def-florian7/salar77h/repos/video_similarity_search/config/custom_configs/slowfast_kinetics_cc.yaml --gpu 0,1,2,3 --num_data_workers 4 --batch_size 40 --output /home/salar77h/projects/def-florian7/salar77h/repos/video_similarity_search/output_kinetics2 --checkpoint_path /home/salar77h/projects/def-florian7/salar77h/repos/video_similarity_search/output_kinetics2/tnet_checkpoints/slowfast/checkpoint.pth.tar
