#!/bin/bash
#SBATCH --account=def-florian7_gpu
#SBATCH --time=1-20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:t4:4
#SBATCH --mem=32G
#SBATCH --job-name=resnet_pre_ucf_kinetics
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
--pretrain_path '/home/cheny257/projects/def-florian7/cheny257/output/pretrain_ucf/save_200.pth' \
--output '/home/cheny257/projects/def-florian7/cheny257/output/kinetics_pre_ucf_2' \
--gpu 0,1,2,3 \
--batch_size 40 \
--num_data_workers 4 \
--n_classes 101
#
# --checkpoint_path '/home/cheny257/projects/def-florian7/cheny257/output/kinetics_pre_ucf/tnet_checkpoints/3dresnet/checkpoint.pth.tar' \
