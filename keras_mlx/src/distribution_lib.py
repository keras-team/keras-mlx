"""Data parallel training across processes with `mlx.core.distributed`.

MLX gives each process one device, so a process is one replica. Variables
stay replicated and the trainer averages the gradients over all processes.
Sharded layouts, used by `ModelParallel`, are not supported.
"""

import json
import os
import tempfile

import mlx.core as mx

from keras.src.random import seed_generator
from keras.src.utils import rng_utils


def list_devices(device_type=None):
    """Return one device per process, the global devices of the group."""
    if device_type is None:
        device_type = "gpu" if mx.is_available(mx.gpu) else "cpu"
    device_type = device_type.lower()
    if device_type not in ("cpu", "gpu"):
        raise ValueError(
            f"Unknown device type for MLX. Received: device_type={device_type}"
        )
    if not mx.is_available(getattr(mx, device_type)):
        return []
    return [f"{device_type}:{i}" for i in range(num_processes())]


def get_device_count(device_type=None):
    """Return the number of devices, one per process."""
    return len(list_devices(device_type))


def initialize(job_addresses=None, num_processes=None, process_id=None):
    """Join the process group.

    Args:
        job_addresses: Optional string, comma separated `host:port` addresses
            of every process in rank order, used to set up the ring backend.
            If `None`, the configuration `mlx.launch` puts in the environment
            is used.
        num_processes: Optional integer, the total number of processes.
        process_id: Optional integer, the rank of the current process.
    """
    if job_addresses and "MLX_HOSTFILE" in os.environ:
        raise ValueError(
            "Pass either `job_addresses` or set `MLX_HOSTFILE`, not both. "
            f"Received: job_addresses={job_addresses}, "
            f"MLX_HOSTFILE={os.environ['MLX_HOSTFILE']}"
        )
    hostfile = None
    backend = "any"
    if job_addresses:
        addresses = job_addresses.split(",")
        if num_processes is not None and len(addresses) != num_processes:
            raise ValueError(
                "The MLX ring backend needs the address of every process in "
                f"`job_addresses`. Received: job_addresses={job_addresses}, "
                f"num_processes={num_processes}"
            )
        with tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False
        ) as f:
            json.dump([[address] for address in addresses], f)
        hostfile = f.name
        os.environ["MLX_HOSTFILE"] = hostfile
        backend = "ring"
    set_rank = process_id is not None and "MLX_RANK" not in os.environ
    if set_rank:
    strict = backend == "ring" or (num_processes is not None and num_processes > 1) or "MLX_HOSTFILE" in os.environ
    try:
        group = mx.distributed.init(strict=strict, backend=backend)

    # A single process runs without a distributed backend.
    strict = "MLX_HOSTFILE" in os.environ
    try:
        group = mx.distributed.init(strict=strict, backend=backend)
    finally:
        # The ring backend reads these once, while connecting.
        if hostfile is not None:
            os.remove(hostfile)
            del os.environ["MLX_HOSTFILE"]
        if set_rank:
            del os.environ["MLX_RANK"]

    if num_processes is not None and group.size() != num_processes:
        raise ValueError(
            f"The MLX distributed group has {group.size()} processes. "
            f"Received: num_processes={num_processes}"
        )
    if process_id is not None and group.rank() != process_id:
        raise ValueError(
            f"The MLX distributed group gave this process rank "
            f"{group.rank()}. Received: process_id={process_id}"
        )
    initialize_rng()


def initialize_rng():
    """Give every process the same global seed, so weights start equal."""
    if rng_utils.get_random_seed() is not None:
        return
    group = mx.distributed.init()
    if group.size() == 1:
        return
    seed = mx.array(seed_generator.make_default_seed(), mx.int32)
    rng_utils.set_random_seed(broadcast(seed, group).item())


def broadcast(x, group):
    """Return process 0's value of `x` on every process of `group`."""
    # MLX has no broadcast, so sum process 0's value with zeros.
    x = x if group.rank() == 0 else mx.zeros_like(x)
    return mx.distributed.all_sum(x, group=group)


def num_processes():
    """Return the number of processes, 1 without a distributed backend."""
    return mx.distributed.init().size()


def process_id():
    """Return the rank of the current process."""
    return mx.distributed.init().rank()


def distribute_tensor(tensor, layout):
    """Return `tensor` unchanged, only replicated layouts are supported."""
    if layout is None:
        return tensor
    if any(axis is not None for axis in layout.axes):
        raise NotImplementedError(
            "The MLX backend only supports replicated layouts, use "
            f"`keras.distribution.DataParallel`. Received: layout={layout}"
        )
    return tensor
