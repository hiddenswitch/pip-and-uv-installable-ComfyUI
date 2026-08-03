from __future__ import annotations

import multiprocessing.connection
import os
import sys

from comfy.distributed.config import resolve_distributed_configuration

from .distributed import _worker_main


def main():
    host, port, authkey = sys.argv[1:]
    distributed = resolve_distributed_configuration(environment=os.environ)
    connection = multiprocessing.connection.Client(
        (host, int(port)),
        authkey=bytes.fromhex(authkey),
    )
    rank = distributed.rank
    connection.send(rank)
    world_size, init_method, load_spec = connection.recv()
    _worker_main(rank, world_size, init_method, load_spec, connection)


if __name__ == "__main__":
    main()
