import matplotlib.pyplot as plt
from core import Game, Visualization, Graph, Controller

from multiprocessing import Process


## ## ## ## ##
def draw_game(game, controller):
    fig1, ax1 = plt.subplots()
    vis = Visualization(game, controller, fig1, ax1)
    # pass


# vis.on_reset(None)
def draw_graphs(game, controller):
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig, ax = plt.subplots()
    graph = Graph(game, controller, fig, ax)

    # graph.controller.train_once(10000)


if __name__ == "__main__":
    game = Game()
    controller = Controller(game)
    controller.train(10000)
    training_record = controller.get_metrics()
    # training_record = game.train_agent(50000)
    Visualization.plot_training(training_record)

    p1 = Process(
        target=draw_graphs,
        args=[
            game,
            controller,
        ],
    )
    p1.start()
    p2 = Process(target=draw_game, args=[game, controller])
    p2.start()
    p1.join()
    p2.join()
