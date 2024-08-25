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
    controller = Controller(game, 1000)

    fig1, ax1 = plt.subplots()
    vis = Visualization(game, controller, fig1, ax1)

    # game = Game()
    # controller = Controller(game, 5000)
    # controller.train(5000)
    # fig1, ax1 = plt.subplots()
    # vis = Visualization(game, controller, fig1, ax1)
    # Visualization.plot_training(controller.get_metrics())
