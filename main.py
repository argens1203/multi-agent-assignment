import matplotlib.pyplot as plt
from core import Game, Visualization, Graph, Controller

from multiprocessing import Process, Queue


def draw_game(game, controller):
    fig1, ax1 = plt.subplots()
    vis = Visualization(game, controller, fig1, ax1)


def draw_graphs(game, controller):
    fig, ax = plt.subplots()
    graph = Graph(controller, fig, ax)


def train_in_parallel(controller, ep):
    return Process(target=controller.train, args=[ep])


if __name__ == "__main__":
    game = Game()
    controller = Controller(game)

    p1 = Process(
        target=draw_graphs,
        args=[
            game,
            controller,
        ],
    )
    p2 = Process(target=draw_game, args=[game, controller])
    p3 = train_in_parallel(controller, 5000)

    controller.train(5000)
    p1.start()
    p3.start()
    p2.start()

    p1.join()
    p2.join()
    p3.join()
