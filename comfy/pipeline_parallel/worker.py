from __future__ import annotations

import multiprocessing.connection
import sys

from .distributed import _worker_main


def main():
    host, port, authkey, rank = sys.argv[1:]
    connection = multiprocessing.connection.Client(
        (host, int(port)),
        authkey=bytes.fromhex(authkey),
    )
    rank = int(rank)
    connection.send(rank)
    world_size, init_method, load_spec = connection.recv()
    _worker_main(rank, world_size, init_method, load_spec, connection)


if __name__ == "__main__":
    main()
