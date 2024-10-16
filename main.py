import matplotlib.pyplot as plt

from core import Agent, Grid, Storage, Visualization, MLGraph
from constants import state_size

if __name__ == "__main__":
    width, height = 4, 4
    max_itr = 1000

    storage = Storage(max_itr)
    agent1 = Agent(
        0,
        state_size,
        [(0, -1), (0, 1), (-1, 0), (1, 0)],
    )
    agent2 = Agent(
        0,
        state_size,
        [(0, -1), (0, 1), (-1, 0), (1, 0)],
    )
    grid = Grid(width, height, [agent1, agent2], storage)
    grid.reset()

    grid.train(2500)
    fig2, ax2 = plt.subplots()
    MLGraph(storage.ml_losses, fig2, ax2).show()

    fig1, ax1 = plt.subplots()
    vis = Visualization(fig1, ax1)
    vis.bind(storage, grid).show()
