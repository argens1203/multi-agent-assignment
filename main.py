import matplotlib.pyplot as plt

from frontend import Visualization, Controller, Model, Storage, Trainer
from core import Agent, Grid
from shared import State

if __name__ == "__main__":
    fig1, ax1 = plt.subplots()
    width, height = 4, 4
    max_itr = 1000

    vis = Visualization(fig1, ax1)
    model = Model()
    controller = Controller()

    storage = Storage(max_itr)
    trainer = Trainer(max_itr)

    agent = Agent(
        0,
        State.get_possible_states(width, height),
        State.get_possible_actions(),
    )
    grid = Grid(width, height)
    grid.add_agent(agent)

    trainer.bind(model, storage, grid, [agent])

    model.set_grid(grid).add_agent(agent).reset()
    controller.bind(model).add_helper(storage, trainer)

    trainer.train(5000)
    vis.bind(model, controller).show()

    # game = Game()
    # controller = Controller(game, 1000)
    # controller.train(1000)
    # fig1, ax1 = plt.subplots()
    # vis = Visualization(game, controller, fig1, ax1)
    # Visualization.plot_training(controller.get_metrics())
