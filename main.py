import matplotlib.pyplot as plt

from frontend import Visualization, Controller, Model, Storage, Trainer
from core import Agent, Grid
from shared import State

if __name__ == "__main__":
    fig1, ax1 = plt.subplots()
    width, height = 5, 5

    vis = Visualization(fig1, ax1)
    model = Model()
    controller = Controller()

    max_itr = 1000
    storage = Storage(max_itr)
    trainer = Trainer(max_itr)

    trainer.bind(model, storage)

    agent = Agent(
        0,
        State.get_possible_states(width, height),
        State.get_possible_actions(),
    )
    grid = Grid(width, height)
    grid.add_agent(agent)

    model.bind(grid).add_agent(agent).reset()
    controller.bind(model, storage, trainer)
    vis.bind(model, controller).show()

    # game = Game()
    # controller = Controller(game, 1000)
    # controller.train(1000)
    # fig1, ax1 = plt.subplots()
    # vis = Visualization(game, controller, fig1, ax1)
    # Visualization.plot_training(controller.get_metrics())
