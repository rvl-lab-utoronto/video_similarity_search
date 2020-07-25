#!/bin/bash
#SBATCH --account=def-florian7_gpu
#SBATCH --time=0-20:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:t4:4
#SBATCH --mem=48G
#SBATCH --job-name=resnet_kinetics
#SBATCH --output=%x-%j.out

cd $SLURM_TMPDIR
mkdir work_kinetics
cd work_kinetics
tar -xzf /home/cheny257/projects/def-florian7/datasets/kinetics400/frames_shortedge320px_25fps/val_split.tar.gz
echo 'Extracted val zip'
tar -xzf /home/cheny257/projects/def-florian7/datasets/kinetics400/frames_shortedge320px_25fps/train_split.tar.gz
echo 'Extracted train zip'


python /home/cheny257/projects/def-florian7/cheny257/code/video_similarity_search/train.py \
--cfg '/home/cheny257/projects/def-florian7/cheny257/code/video_similarity_search/config/custom_configs/cc_resnet_kinetics.yaml' \
--output '/home/cheny257/projects/def-florian7/cheny257/output/ResNet18_K' \
--gpu 0,1,2,3 \
--batch_size 40 \
--num_data_workers 4
# --checkpoint_path '/home/cheny257/projects/def-florian7/cheny257/output/kinetics_4/tnet_checkpoints/3dresnet/checkpoint.pth.tar' \
