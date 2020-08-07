#!/bin/bash
#SBATCH --account=def-florian7_gpu 
#SBATCH --time=0-08:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=slowfast_kinetics
#SBATCH --output=%x-%j.out
#SBATCH --gres=gpu:v100l:4
#SBATCH --mem=48G
#SBATCH --cpus-per-task=10
# --wait-all-nodes=1

clush -w $(slurm_hl2hl.py --format PDSH) tar -xzf /home/salar77h/projects/def-florian7/datasets/kinetics400/frames_shortedge320px_25fps/val_split.tar.gz -C $SLURM_TMPDIR
echo 'Extracted val zip'
clush -w $(slurm_hl2hl.py --format PDSH) tar -xzf /home/salar77h/projects/def-florian7/datasets/kinetics400/frames_shortedge320px_25fps/train_split.tar.gz -C $SLURM_TMPDIR
echo 'Extracted train zip'

cd $SLURM_TMPDIR

export MASTER_ADDRESS=$(hostname)
echo $MASTER_ADDRESS

#MPORT=`ss -tan | awk '{print $4}' | cut -d':' -f2 | \
#       grep "[2-9][0-9]\{3,3\}" | grep -v "[0-9]\{5,5\}" | \
#       sort | uniq | shuf`
#
MPORT=3456
echo $MPORT

srun python /home/salar77h/projects/def-florian7/salar77h/repos/video_similarity_search/train.py --cfg /home/salar77h/projects/def-florian7/salar77h/repos/video_similarity_search/config/custom_configs/slowfast_kinetics_cc.yaml --gpu 0,1,2,3 --num_data_workers 4 --batch_size 40 --output /home/salar77h/projects/def-florian7/salar77h/repos/video_similarity_search/output_kin_temp --num_shards 2 --epoch 2 --ip_address_port tcp://$MASTER_ADDRESS:$MPORT --compute_canada

