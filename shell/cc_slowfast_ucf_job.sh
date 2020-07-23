#!/bin/bash
#SBATCH --account=def-florian7_gpu 
#SBATCH --time=0-01:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:t4:4
#SBATCH --mem=48G
#SBATCH --job-name=slowfast_ucf
#SBATCH --output=%x-%j.out
#SBATCH --wait-all-nodes=1

#srun /home/salar77h/projects/def-florian7/salar77h/repos/video_similarity_search/shell/get_node_id

#srun cd $SLURM_TMPDIR
#srun mkdir work_ucf
#srun cd work_ucf
#srun tar -xzf /home/salar77h/projects/def-florian7/datasets/UCF101/jpg.tar.gz
#echo 'Extracted jpg.tar.gz'

echo $SLURM_NODE_ALIASES

srun -N1 -n1 -r 1 python /home/salar77h/projects/def-florian7/salar77h/repos/video_similarity_search/train.py --cfg /home/salar77h/projects/def-florian7/salar77h/repos/video_similarity_search/config/custom_configs/slowfast_ucf_cc.yaml --gpu 0,1,2,3 --num_data_workers 4 --batch_size 40 --output /home/salar77h/projects/def-florian7/salar77h/repos/video_similarity_search/output_ucf1-distrib --num_shards 2 --shard_id 0 --epoch 1 &

srun -N1 -n1 -r 0 python /home/salar77h/projects/def-florian7/salar77h/repos/video_similarity_search/train.py --cfg /home/salar77h/projects/def-florian7/salar77h/repos/video_similarity_search/config/custom_configs/slowfast_ucf_cc.yaml --gpu 0,1,2,3 --num_data_workers 4 --batch_size 40 --output /home/salar77h/projects/def-florian7/salar77h/repos/video_similarity_search/output_ucf1-distrib --num_shards 2 --shard_id 1 --epoch 1 &

wait
