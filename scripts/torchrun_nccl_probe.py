import os
import time
import argparse

import torch
import torch.distributed as dist


def main():
    # Set fallback environment variables if not running with torchrun
    if "WORLD_SIZE" not in os.environ:
        os.environ["WORLD_SIZE"] = "1"
    if "RANK" not in os.environ:
        os.environ["RANK"] = "0"
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "29500"

    parser = argparse.ArgumentParser(description="Minimal torchrun NCCL/Gloo probe")
    parser.add_argument("--numel", type=int, default=1, help="Number of elements in test tensor")
    args = parser.parse_args()

    has_cuda = torch.cuda.is_available()
    backend = "nccl" if has_cuda else "gloo"

    # Set device if CUDA
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if has_cuda:
        torch.cuda.set_device(local_rank)

    # Init process group
    dist.init_process_group(backend=backend)

    rank = dist.get_rank()
    world = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}") if has_cuda else torch.device("cpu")

    print(f"[Probe] Backend={backend} | Rank {rank}/{world} | Device={device}", flush=True)

    # Small all_reduce
    t = torch.ones(args.numel, device=device, dtype=torch.int64) * rank
    t0 = time.perf_counter()
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t1 = time.perf_counter()
    print(f"[Probe] all_reduce sum={int(t[0].item())} in {t1 - t0:.4f}s", flush=True)

    # Barrier
    t2 = time.perf_counter()
    if backend == "nccl" and has_cuda:
        dist.barrier(device_ids=[torch.cuda.current_device()])
    else:
        dist.barrier()
    t3 = time.perf_counter()
    print(f"[Probe] barrier done in {t3 - t2:.4f}s", flush=True)

    # all_gather
    val = torch.tensor([rank], device=device, dtype=torch.int64)
    out = [torch.empty_like(val) for _ in range(world)]
    t4 = time.perf_counter()
    dist.all_gather(out, val)
    t5 = time.perf_counter()
    gathered = [int(x.item()) for x in out]
    print(f"[Probe] all_gather ranks={gathered} in {t5 - t4:.4f}s", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()


