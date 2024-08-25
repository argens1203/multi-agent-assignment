import matplotlib.pyplot as plt
from core import Game, Visualization, Graph, Controller

from multiprocessing import Process, Queue


def draw_game(game, controller):
    fig1, ax1 = plt.subplots()
    vis = Visualization(game, controller, fig1, ax1)


def draw_graphs(game, controller):
    fig, ax = plt.subplots()
    graph = Graph(controller, fig, ax)


def train(controller, ep):
    controller.train(ep)


if __name__ == "__main__":
    game = Game()
    controller = Controller(game)

    # tps = []
    # gps = []
    # for i in range(100):

    # graph_p = Process(
    #     target=draw_graphs,
    #     args=[
    #         game,
    #         controller,
    #     ],
    # )
    # train_p = Process(target=train, args=[controller, 5000])
    # tps.append(train_p)
    # gps.append(graph_p)

    fig1, ax1 = plt.subplots()
    vis = Visualization(game, controller, fig1, ax1)

    # game_p = Process(target=draw_game, args=[game, controller])

    # controller.train(5000)
    # graph_p.start()
    # train_p.start()
    # # graph_p.join()
    # train_p.join()
    # game_p.start()

    # game_p.join()
