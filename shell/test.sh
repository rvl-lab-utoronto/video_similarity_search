#!/bin/bash
#SBATCH --account=def-florian7_gpu
#SBATCH --time=0-01:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:t4:4
#SBATCH --mem=48G
#SBATCH --job-name=test
#SBATCH --output=%x-%j.out


nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)  # Getting the node names
nodes_array=( $nodes )
NUM_NODES=${#ArrayName[@]}
echo number of node:$NUM_NODES

node1=${nodes_array[0]}
echo node:$node1
# ip_address=$(python -c "import socket; print(socket.gethostbyname(socket.gethostname()))")
ip_address=$(python -c "import ray.services; print(ray.services.get_node_ip_address())")
port='6379'
ip_address_port=tcp://${ip_address}:${port}
echo ip_address_port:${ip_address_port}

# cd $SLURM_TMPDIR
# mkdir work_kinetics
# cd work_kinetics
# tar -xzf /home/cheny257/projects/def-florian7/datasets/kinetics400/frames_shortedge320px_25fps/val_split.tar.gz
# echo 'Extracted val zip'
# tar -xzf /home/cheny257/projects/def-florian7/datasets/kinetics400/frames_shortedge320px_25fps/train_split.tar.gz
# echo 'Extracted train zip'

echo trying out srun...

srun --nodes=1 --ntasks=1 -w $node1 \
python /home/cheny257/projects/def-florian7/cheny257/code/video_similarity_search/train.py \
--cfg '/home/cheny257/projects/def-florian7/cheny257/code/video_similarity_search/config/custom_configs/cc_resnet_kinetics.yaml' \
--output '/home/cheny257/projects/def-florian7/cheny257/output/kinetics_4' \
--checkpoint_path '/home/cheny257/projects/def-florian7/cheny257/output/kinetics_4/tnet_checkpoints/3dresnet/checkpoint.pth.tar' \
--gpu 0,1,2,3 \
--batch_size 40 \
--num_data_workers 4 \
--num_shards $NUM_NODES \
--shard_id 0 \
--ip_address_port $ip_address_port \
--epoch 1 &

sleep 5
for ((  i=1; i<$NUM_NODES; i++ ))
do
  node2=${nodes_array[$i]}
  echo node:$node2
  srun --nodes=1 --ntasks=1 -w $node2 \
  python /home/cheny257/projects/def-florian7/cheny257/code/video_similarity_search/train.py \
  --cfg '/home/cheny257/projects/def-florian7/cheny257/code/video_similarity_search/config/custom_configs/cc_resnet_kinetics.yaml' \
  --output '/home/cheny257/projects/def-florian7/cheny257/output/kinetics_4' \
  --checkpoint_path '/home/cheny257/projects/def-florian7/cheny257/output/kinetics_4/tnet_checkpoints/3dresnet/checkpoint.pth.tar' \
  --gpu 0,1,2,3 \
  --batch_size 40 \
  --num_data_workers 4 \
  --num_shards $NUM_NODES \
  --shard_id $i \
  --ip_address_port $ip_address_port \
  --epoch 1 &
  sleep 5
done

wait
echo done launching
