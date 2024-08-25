import matplotlib.pyplot as plt
from core import Game, Visualization, Graph

from multiprocessing import Process


## ## ## ## ##
def draw_game(game):
    fig1, ax1 = plt.subplots()
    vis = Visualization(game, fig1, ax1)
    # pass


# vis.on_reset(None)
def draw_graphs(game):
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    # graph = Graph(game, fig, (ax1, ax2))
    fig, ax = plt.subplots()
    graph = Graph(game, fig, ax)

    # graph.controller.train_once(10000)


if __name__ == "__main__":
    game = Game()
    training_record = game.train_agent(1000)
    # Visualization.plot_training(training_record)

    p1 = Process(
        target=draw_graphs,
        args=[
            game,
        ],
    )
    p1.start()
    p2 = Process(target=draw_game, args=[game])
    p2.start()
    p1.join()
    p2.join()
