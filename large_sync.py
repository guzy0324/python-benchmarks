"""
场景：用多卡在一个数据上做推理，之后对推理的结果进行同步。最简单的方式是每个卡上的结果都存到显存上，等整个数据集都推理完之后再进行一次同步。但是问题是显存有限，如果结果数据规模大会导致OOM。把结果存到内存中是一种解决思路，但是有两个方案，不确定哪个更快。

方案：
1. 在GPU上同步每个batch的结果，再存到内存中；这样的好处是GPU同步速度快，坏处是同步次数多。
2. 将每个batch的结果存到内存中，再在CPU上同步所有batch的结果；这样的好处是同步次数仅一次，坏处是CPU同步速度慢。
"""

import builtins
import random
import warnings
from collections import defaultdict
from time import perf_counter

import numpy as np
import seaborn as sns
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from matplotlib import pyplot as plt


@torch.no_grad()
def concat_all_gather(tensor, world_size, group=None, async_op=False):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    if world_size == 1:
        return tensor
    tensors_gather = [torch.ones_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensors_gather, tensor, group=group, async_op=async_op)

    output = torch.cat(tensors_gather, dim=0)
    return output


def set_seed(seed=42, global_rank=0):
    # ! https://github.com/facebookresearch/deit/issues/150#issue-1158078425
    random.seed(seed + global_rank)
    np.random.seed(seed + global_rank)
    torch.manual_seed(seed + global_rank)
    # ! https://github.com/Lightning-AI/pytorch-lightning/issues/19033
    # torch.cuda.manual_seed(seed + global_rank)
    # torch.cuda.manual_seed_all(seed + global_rank)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Timeit:
    avg_time_dict = defaultdict(AverageMeter)

    def __init__(self, name, log_interval=1):
        self.name = name
        self.log_interval = log_interval

    def __enter__(self):
        self.start = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        torch.cuda.synchronize()
        self.time = perf_counter() - self.start
        Timeit.avg_time_dict[self.name].update(self.time)
        if Timeit.avg_time_dict[self.name].count % self.log_interval == 0:
            self.readout = f"{self.name}: {self.time:.3f} ({Timeit.avg_time_dict[self.name].avg:.3f}) seconds"
            print(self.readout)


def worker(rank, world_size, dataset_size_list, batch_size, shape, warmup, trial):
    print(f"rank: {rank}, world_size: {world_size}")
    if rank != 0:

        def print_pass(*args, **kwargs):
            pass

        builtins.print = print_pass
        warnings.warn = print_pass

    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://localhost:10001",
        world_size=world_size,
        rank=rank,
    )
    dist.barrier()
    group_gloo = dist.new_group(list(range(world_size)), backend="gloo")
    dist.barrier(group_gloo)

    gpu_time_list = []
    cpu_time_list = []
    for dataset_size in dataset_size_list:
        dataset_size_per_gpu = dataset_size // world_size
        batch_size_per_gpu = batch_size // world_size
        num_steps = dataset_size_per_gpu // batch_size_per_gpu

        set_seed(global_rank=rank)
        for i in range(warmup + trial):
            with Timeit("warmup gpu" if i < warmup else "gpu"):
                if rank == 0:
                    output_list = []
                for j in range(num_steps):
                    output = torch.randn(batch_size_per_gpu, shape).cuda()
                    output = concat_all_gather(output, world_size)
                    if rank == 0:
                        output_list.append(output.cpu())

        set_seed(global_rank=rank)
        for i in range(trial):
            with Timeit("warmup cpu" if i < warmup else "cpu"):
                if rank == 0:
                    output_list = []
                for j in range(num_steps):
                    output = torch.randn(batch_size_per_gpu, shape).cuda()
                    output = concat_all_gather(
                        output.cpu(), world_size, group=group_gloo
                    )
                    if rank == 0:
                        output_list.append(output)

        gpu_time_list.append(Timeit.avg_time_dict["gpu"].avg)
        cpu_time_list.append(Timeit.avg_time_dict["cpu"].avg)
        Timeit.avg_time_dict.clear()

    if rank == 0:
        plt.figure()
        sns.lineplot(x=dataset_size_list, y=gpu_time_list, label="gpu")
        sns.lineplot(x=dataset_size_list, y=cpu_time_list, label="cpu")
        plt.savefig("large_sync.png")
        plt.close()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    # dataset_size = 1281167 + 50000
    dataset_size_list = [50000, 100000, 200000, 500000]
    batch_size = 1024
    shape = 2048
    warmup = 5
    trial = 10
    mp.spawn(
        worker,
        nprocs=world_size,
        args=(world_size, dataset_size_list, batch_size, shape, warmup, trial),
    )
