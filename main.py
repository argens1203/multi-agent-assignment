import matplotlib.pyplot as plt
from core import Game, Visualization, Graph

from multiprocessing import Process


def f(name):
    print("hello", name)


## ## ## ## ##
def draw_game(game):
    fig1, ax1 = plt.subplots()
    vis = Visualization(game, fig1, ax1)


# vis.on_reset(None)
def draw_graphs(game):
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    # graph = Graph(game, fig, (ax1, ax2))
    fig, ax = plt.subplots()
    graph = Graph(game, fig, ax)


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
    draw_game(game)
    # p2 = Process(target=draw_two)
    p1.join()
