import matplotlib.pyplot as plt

from core import Agent1, Agent2, Grid, Storage, Visualization, MLGraph
from constants import state_size

if __name__ == "__main__":
    width, height = 4, 4
    max_itr = 1000

    storage = Storage(max_itr)
    possible_actions = [(0, -1), (0, 1), (-1, 0), (1, 0)]

    grid = Grid(
        width,
        height,
        [
            Agent1(0, state_size, possible_actions),
            Agent1(0, state_size, possible_actions),
            Agent2(0, state_size, possible_actions),
            Agent2(0, state_size, possible_actions),
        ],
        storage,
    )
    grid.reset()

    grid.train(2500)
    fig2, ax2 = plt.subplots()
    MLGraph(storage.ml_losses, fig2, ax2).show()

    fig1, ax1 = plt.subplots()
    vis = Visualization(fig1, ax1)
    vis.bind(storage, grid).show()
