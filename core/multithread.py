import matplotlib.pyplot as plt
import numpy as np

from typing import TYPE_CHECKING
from multiprocessing import Process, shared_memory, Pipe

from .view import Graph, TestGraph

if TYPE_CHECKING:
    from .storage import Storage
    from .grid import Grid


def draw_graphs(storage: "Storage"):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    graph = Graph(storage, fig, axs)


def train(grid: "Grid", connection, ep):
    grid.train(ep)
    # TODO: remove hardcode
    q = grid.agents[0].get_q_table()

    shm = shared_memory.SharedMemory(create=True, size=q.nbytes)
    b = np.ndarray(q.shape, dtype=q.dtype, buffer=shm.buf)
    b[:] = q[:]
    connection.send(shm.name)
    shm.close()


def get_process(storage: "Storage", grid: "Grid"):
    conn1, conn2 = Pipe()
    graph_p = Process(
        target=draw_graphs,
        args=[
            storage,
        ],
    )
    # TODO: remove hardcode
    train_p = Process(target=train, args=[grid, conn2, 1000])
    return graph_p, train_p, conn1


def test(grid: "Grid", ep):
    grid.test(ep)


def draw_test_graph(storage: "Storage"):
    fig, axs = plt.subplots()
    graph = TestGraph(storage, fig, axs)


def get_test_process(storage: "Storage", grid: "Grid", ep=1000):
    graph_p = Process(
        target=draw_test_graph,
        args=[storage],
    )
    test_p = Process(target=test, args=[grid, ep])
    return graph_p, test_p


def get_np_from_name(name):
    existing_shm = shared_memory.SharedMemory(name=name)
    # TODO: remove hardcode
    q = np.ndarray((5**5, 4), buffer=existing_shm.buf)
    s = np.copy(q)
    existing_shm.close()
    existing_shm.unlink()
    return s
