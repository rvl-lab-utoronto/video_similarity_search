#!/bin/bash
#SBATCH --account=def-florian7_gpu 
#SBATCH --time=0-00:30:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=slowfast_ucf
#SBATCH --output=%x-%j.out
#SBATCH --gres=gpu:t4:4
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
# --wait-all-nodes=1

#srun /home/salar77h/projects/def-florian7/salar77h/repos/video_similarity_search/shell/get_node_id

#srun cd $SLURM_TMPDIR
#srun mkdir work_ucf
#srun cd work_ucf
#srun tar -xzf /home/salar77h/projects/def-florian7/datasets/UCF101/jpg.tar.gz
#echo 'Extracted jpg.tar.gz'

#echo $SLURM_NODE_ALIASES
#echo $SLURM_SRUN_COMM_HOST

#We want names of master and slave nodes
#MASTER=`/bin/hostname -s`
#SLAVES=`scontrol show hostnames $SLURM_JOB_NODELIST | grep -v $MASTER`
#Make sure this node (MASTER) comes first
#HOSTLIST="$MASTER $SLAVES"
#echo $HOSTLIST

export MASTER_ADDRESS=$(hostname)
echo $MASTER_ADDRESS

#MPORT=`ss -tan | awk '{print $4}' | cut -d':' -f2 | \
#       grep "[2-9][0-9]\{3,3\}" | grep -v "[0-9]\{5,5\}" | \
#       sort | uniq | shuf`
#
MPORT=3456
echo $MPORT

#srun -N1 -n1 -r 1 cd $SLURM_TMPDIR && mkdir work_ucf && cd work_ucf && tar -xzf /home/salar77h/projects/def-florian7/datasets/UCF101/jpg.tar.gz && 
srun python /home/salar77h/projects/def-florian7/salar77h/repos/video_similarity_search/train.py --cfg /home/salar77h/projects/def-florian7/salar77h/repos/video_similarity_search/config/custom_configs/slowfast_ucf_cc.yaml --gpu 0 --num_data_workers 4 --batch_size 40 --output /home/salar77h/projects/def-florian7/salar77h/repos/video_similarity_search/output_ucf1-distrib --num_shards 2 --epoch 2 --ip_address_port tcp://$MASTER_ADDRESS:$MPORT --compute_canada

#srun -N1 -n1 -r 0 cd $SLURM_TMPDIR && mkdir work_ucf && cd work_ucf && tar -xzf /home/salar77h/projects/def-florian7/datasets/UCF101/jpg.tar.gz python /home/salar77h/projects/def-florian7/salar77h/repos/video_similarity_search/train.py --cfg /home/salar77h/projects/def-florian7/salar77h/repos/video_similarity_search/config/custom_configs/slowfast_ucf_cc.yaml --gpu 0 --num_data_workers 4 --batch_size 40 --output /home/salar77h/projects/def-florian7/salar77h/repos/video_similarity_search/output_ucf1-distrib --num_shards 2 --shard_id 1 --epoch 2 --ip_address_port tcp://$MASTER_ADDRESS:$MPORT &

#wait
