import matplotlib.pyplot as plt
import numpy as np

from typing import TYPE_CHECKING
from multiprocessing import Process, shared_memory, Pipe

from .v_graph import Graph, TestGraph

if TYPE_CHECKING:
    from .controller import Controller, Storage, Trainer
    from .model import Model


def draw_graphs(storage: "Storage"):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    graph = Graph(storage, fig, axs)


def train(trainer: "Trainer", connection, ep):
    trainer.train(ep)
    q = trainer.model.agent[0].get_q_table()

    shm = shared_memory.SharedMemory(create=True, size=q.nbytes)
    b = np.ndarray(q.shape, dtype=q.dtype, buffer=shm.buf)
    b[:] = q[:]
    connection.send(shm.name)
    shm.close()


def get_process(storage: "Storage", trainer: "Trainer"):
    conn1, conn2 = Pipe()
    graph_p = Process(
        target=draw_graphs,
        args=[
            storage,
        ],
    )
    train_p = Process(target=train, args=[trainer, conn2, 1000])
    return graph_p, train_p, conn1


def test(trainer: "Trainer", ep):
    trainer.test(ep)


def draw_test_graph(storage: "Storage"):
    fig, axs = plt.subplots()
    graph = TestGraph(storage, fig, axs)


def get_test_process(storage: "Storage", trainer: "Trainer", ep=1000):
    graph_p = Process(
        target=draw_test_graph,
        args=[storage],
    )
    test_p = Process(target=test, args=[trainer, ep])
    return graph_p, test_p


def get_np_from_name(name):
    existing_shm = shared_memory.SharedMemory(name=name)
    q = np.ndarray((5**5, 4), buffer=existing_shm.buf)
    s = np.copy(q)
    existing_shm.close()
    existing_shm.unlink()
    return s
