#!/bin/bash
#SBATCH --account=def-florian7_gpu 
#SBATCH --time=1-00:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=res_ucf_atten
#SBATCH --output=%x-%j.out
#SBATCH --gres=gpu:v100l:4
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
# --wait-all-nodes=1

clush -w $(slurm_hl2hl.py --format PDSH) tar -xzf /home/salar77h/projects/def-florian7/datasets/UCF101/poolnet_new.tar.gz -C $SLURM_TMPDIR
echo 'Extracted mask zip'
clush -w $(slurm_hl2hl.py --format PDSH) tar -xzf /home/salar77h/projects/def-florian7/datasets/UCF101/jpg.tar.gz -C $SLURM_TMPDIR
echo 'Extracted rgb zip'

cd $SLURM_TMPDIR

export MASTER_ADDRESS=$(hostname)
echo $MASTER_ADDRESS

#MPORT=`ss -tan | awk '{print $4}' | cut -d':' -f2 | \
#       grep "[2-9][0-9]\{3,3\}" | grep -v "[0-9]\{5,5\}" | \
#       sort | uniq | shuf`
#
MPORT=3456
echo $MPORT

srun python /home/salar77h/projects/def-florian7/salar77h/repos/video_similarity_search/online_train.py --iterative_cluster --cfg /home/salar77h/projects/def-florian7/salar77h/repos/video_similarity_search/config/custom_configs/resnet_ucf_itercluster_mask_cc.yaml --gpu 0,1,2,3 --num_data_workers 4 --batch_size 100 --output /home/salar77h/projects/def-florian7/salar77h/repos/video_similarity_search/output_ucf15 --num_shards 2 --ip_address_port tcp://$MASTER_ADDRESS:$MPORT --compute_canada --epoch 801 
#--checkpoint_path /home/salar77h/projects/def-florian7/salar77h/repos/video_similarity_search/output_ucf14-itercluster-4d-bn-attention-new/tnet_checkpoints/3dresnet/checkpoint.pth.tar