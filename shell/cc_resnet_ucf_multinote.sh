#!/bin/bash
#SBATCH --account=def-florian7_gpu
#SBATCH --time=0-00:10:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=resnet_ucf_multinodes
#SBATCH --output=%x-%j.out
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
# --wait-all-nodes=1


#echo $SLURM_NODE_ALIASES
#echo $SLURM_SRUN_COMM_HOST

#We want names of master and slave nodes
#MASTER=`/bin/hostname -s`
#SLAVES=`scontrol show hostnames $SLURM_JOB_NODELIST | grep -v $MASTER`
#Make sure this node (MASTER) comes first
#HOSTLIST="$MASTER $SLAVES"
#echo $HOSTLIST


module load python/3.6

source /home/cheny257/projects/def-florian7/cheny257/code/resnet_env/bin/activate


which pdsh
pdsh -w $(slurm_hl2hl.py --format PDSH) tar -xvf /home/cheny257/projects/def-florian7/datasets/UCF101/jpg.tar.gz -C $SLURM_TMPDIR
echo 'Extracted jpg.tar.gz'


export MASTER_ADDRESS=$(hostname)
echo master_address:$MASTER_ADDRESS

MPORT=3456
echo master_port:$MPORT

srun python /home/cheny257/projects/def-florian7/cheny257/code/video_similarity_search/train.py \
--cfg '/home/cheny257/projects/def-florian7/cheny257/code/video_similarity_search/config/custom_configs/cc_resnet_ucf.yaml' \
--gpu 0 \
--num_data_workers 4 \
--batch_size 40 \
--output '/home/cheny257/projects/def-florian7/cheny257/output/multinodes_test' \
--num_shards 2 \
--epoch 2 \
--ip_address_port tcp://$MASTER_ADDRESS:$MPORT \
-cc

#wait
