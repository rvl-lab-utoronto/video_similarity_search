import torch


# Run function from a child process (this process has its own gpu)
# proc_init_method includes TCP or shared file-system
# NOTE: ensure that the batch normalization used by the model architecture
#       supports this distributed training
def run_process(local_rank_proc, NUM_PROC_PER_SHARD, func, shard_id, NUM_SHARDS, cmd_args, cfg, proc_init_method="tcp://localhost:9999", dist_backend="nccl"):
    WORLD_SIZE = NUM_PROC_PER_SHARD * NUM_SHARDS
    rank_proc = shard_id * NUM_PROC_PER_SHARD + local_rank_proc

    # Operate on a single GPU in current process
    torch.cuda.set_device(local_rank_proc)

    try:
        torch.distributed.init_process_group(backend=dist_backend,
                                             init_method=proc_init_method,
                                             world_size=WORLD_SIZE,
                                             rank=rank_proc)
        print('Initialized gpu process:', rank_proc)

    except Exception as e:
        print('Failed due to:{}'.format(e))
        raise e

    func(cmd_args, cfg)


# Spawn a process for all gpus (across all nodes)
def launch_processes(cmd_args, cfg, func, shard_id, NUM_SHARDS, ip_address_port):
    if cfg.NUM_GPUS > 1:
        torch.multiprocessing.spawn(fn=run_process,
                                    nprocs=cfg.NUM_GPUS,
                                    args=(cfg.NUM_GPUS, func, shard_id, NUM_SHARDS, cmd_args, cfg, ip_address_port)
                                    )
    else:
        func(cmd_args, cfg)


# Reduce tensors across all devices, average the value by default
def all_reduce(tensors, avg=True):
    for tensor in tensors:
        torch.distributed.all_reduce(tensor)
    if avg:
        world_size = torch.distributed.get_world_size()
        for tensor in tensors:
            tensor.mul_(1.0 / world_size)
    return tensors


# Gathers the specified tensors from all processes across machines
def all_gather(tensors):
    gather_list = []
    output_tensor = []
    world_size = torch.distributed.get_world_size()
    for tensor in tensors:
        tensor_placeholder = [
            torch.ones_like(tensor) for _ in range(world_size)
        ]
        torch.distributed.all_gather(tensor_placeholder, tensor, async_op=False)
        gather_list.append(tensor_placeholder)
    for gathered_tensor in gather_list:
        output_tensor.append(torch.cat(gathered_tensor, dim=0))
    return output_tensor


# Determines if the current process is the master process
def is_master_proc(num_gpus):
    if torch.distributed.is_initialized():
        #return torch.distributed.get_rank() % num_gpus == 0
        return torch.distributed.get_rank() == 0
    else:
        return True


# Get the world size
def get_world_size():
    if not torch.distributed.is_available():
        return 1
    if not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size()
