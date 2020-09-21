#!/bin/bash
#SBATCH --account=def-florian7_gpu 
#SBATCH --time=0-23:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=slowfast_ucf
#SBATCH --output=%x-%j.out
#SBATCH --gres=gpu:v100l:4
#SBATCH --mem=32G
#SBATCH --cpus-per-task=10
# --wait-all-nodes=1

clush -w $(slurm_hl2hl.py --format PDSH) tar -xzf /home/salar77h/projects/def-florian7/datasets/UCF101/jpg.tar.gz -C $SLURM_TMPDIR

cd $SLURM_TMPDIR

export MASTER_ADDRESS=$(hostname)
echo $MASTER_ADDRESS

#MPORT=`ss -tan | awk '{print $4}' | cut -d':' -f2 | \
#       grep "[2-9][0-9]\{3,3\}" | grep -v "[0-9]\{5,5\}" | \
#       sort | uniq | shuf`
#
MPORT=3456
echo $MPORT

srun python /home/salar77h/projects/def-florian7/salar77h/repos/video_similarity_search/online_train.py --cfg /home/salar77h/projects/def-florian7/salar77h/repos/video_similarity_search/config/custom_configs/slowfast_ucf_cc.yaml --gpu 0,1,2,3 --num_data_workers 4 --batch_size 40 --output /home/salar77h/projects/def-florian7/salar77h/repos/video_similarity_search/output_ucf6-clusters-res-p20 --num_shards 2 --ip_address_port tcp://$MASTER_ADDRESS:$MPORT --compute_canada MODEL.ARCH 3dresnet DATASET.POSITIVE_SAMPLING_P 0.2
