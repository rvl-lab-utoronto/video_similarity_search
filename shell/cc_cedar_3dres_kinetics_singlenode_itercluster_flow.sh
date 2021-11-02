#!/bin/bash
#SBATCH --account=def-florian7_gpu 
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=kin_flow_r18
#SBATCH --output=%x-%j.out
#SBATCH --gres=gpu:p100l:4
#SBATCH --mem=100G
#SBATCH --cpus-per-task=24
# --wait-all-nodes=1

clush -w $(slurm_hl2hl.py --format PDSH) tar -xzf /home/salar77h/projects/def-florian7/datasets/kinetics400/frames_shortedge320px_25fps/val_split.tar.gz -C $SLURM_TMPDIR
echo 'Extracted val zip'
clush -w $(slurm_hl2hl.py --format PDSH) tar -xzf /home/salar77h/projects/def-florian7/datasets/kinetics400/frames_shortedge320px_25fps/train_split.tar.gz -C $SLURM_TMPDIR
echo 'Extracted train zip'

#flow
clush -w $(slurm_hl2hl.py --format PDSH) tar -xzf /home/salar77h/projects/def-florian7/datasets/kinetics400/FLOW/FLOW-u-jpg.tar.gz -C $SLURM_TMPDIR
echo 'Extracted flow train zip'

cd $SLURM_TMPDIR

#export MASTER_ADDRESS=$(hostname)
#echo $MASTER_ADDRESS

#MPORT=3456
#echo $MPORT

ROOTDIR=/home/salar77h/projects/def-florian7/salar77h/repos/video_similarity_search 

#mkdir $ROOTDIR/output_ucf15-4gpu-s3d-4
#cp $ROOTDIR/output_ucf15-4gpu-s3d-3/vid_clusters.txt $ROOTDIR/output_ucf15-4gpu-s3d-4/

python $ROOTDIR/online_train.py --iterative_cluster --cfg $ROOTDIR/config/custom_configs/resnet_kinetics_itercluster_flow_cc.yaml --gpu 0,1,2,3 --num_data_workers 4 --batch_size 64 --output $ROOTDIR/output_ucf25-kinetics-iterclus-flow-single2 --epoch 601 --checkpoint_path $ROOTDIR/output_ucf25-kinetics-iterclus-flow-single2/tnet_checkpoints/3dresnet/checkpoint.pth.tar VAL.BATCH_SIZE 80 LOSS.LOCAL_LOCAL_CONTRAST True DATA.INPUT_CHANNEL_NUM 3 ITERCLUSTER.METHOD finch
#--checkpoint_path $ROOTDIR/output_ucf16-adam-32/tnet_checkpoints/3dresnet/checkpoint.pth.tar
