import matplotlib.pyplot as plt
import numpy as np

from typing import TYPE_CHECKING
from multiprocessing import Process, shared_memory, Pipe

from .view_graph import Graph, TestGraph

if TYPE_CHECKING:
    from .controller import Controller
    from .model import Model


def draw_graphs(game: "Model", controller: "Controller"):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    graph = Graph(controller, fig, axs)


def train(controller: "Controller", connection, ep):
    controller.train(ep)
    q = controller.model.agent[0].get_q_table()

    shm = shared_memory.SharedMemory(create=True, size=q.nbytes)
    b = np.ndarray(q.shape, dtype=q.dtype, buffer=shm.buf)
    b[:] = q[:]
    connection.send(shm.name)
    shm.close()


def get_process(game: "Model", controller: "Controller"):
    conn1, conn2 = Pipe()
    graph_p = Process(
        target=draw_graphs,
        args=[
            game,
            controller,
        ],
    )
    train_p = Process(target=train, args=[controller, conn2, 1000])
    return graph_p, train_p, conn1


def test(controller: "Controller", ep):
    controller.test(ep)


def draw_test_graph(controller: "Controller"):
    fig, axs = plt.subplots()
    graph = TestGraph(controller, fig, axs)


def get_test_process(controller: "Controller"):
    graph_p = Process(
        target=draw_test_graph,
        args=[
            controller,
        ],
    )
    test_p = Process(target=test, args=[controller, 1000])
    return graph_p, test_p


def get_np_from_name(name):
    existing_shm = shared_memory.SharedMemory(name=name)
    q = np.ndarray((5**5, 4), buffer=existing_shm.buf)
    s = np.copy(q)
    existing_shm.close()
    existing_shm.unlink()
    return s
