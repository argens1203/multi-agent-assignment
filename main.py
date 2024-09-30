import matplotlib.pyplot as plt

from frontend import Visualization, Controller, Storage, Trainer, MLGraph
from core import Agent, Grid
from constants import state_size

if __name__ == "__main__":
    width, height = 4, 4
    max_itr = 1000

    controller = Controller()

    storage = Storage(max_itr)
    trainer = Trainer(max_itr)

    agent = Agent(
        0,
        state_size,
        [(0, -1), (0, 1), (-1, 0), (1, 0)],
    )
    grid = Grid(width, height)
    grid.add_agent(agent)

    trainer.bind(storage, grid, [agent])

    grid.reset()
    controller.bind(grid).add_helper(storage, trainer)

    trainer.train(2500)
    fig2, ax2 = plt.subplots()
    MLGraph(storage.ml_losses, fig2, ax2).show()

    fig1, ax1 = plt.subplots()
    vis = Visualization(fig1, ax1)
    vis.bind(controller, storage, grid).show()

    # game = Game()
    # controller = Controller(game, 1000)
    # controller.train(1000)
    # fig1, ax1 = plt.subplots()
    # vis = Visualization(game, controller, fig1, ax1)
    # Visualization.plot_training(controller.get_metrics())
