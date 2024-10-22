import matplotlib.pyplot as plt

from core import Agent1, Agent2, Grid, Storage, Visualization, MLGraph, ExpBuffer, DQN
from constants import state_size, action_size

if __name__ == "__main__":
    width, height = 4, 4
    max_itr = 1000

    storage = Storage(max_itr)
    possible_actions = [(0, -1), (0, 1), (-1, 0), (1, 0), (0, 0)]

    buffer1 = ExpBuffer()
    buffer2 = ExpBuffer()
    dqn1 = DQN(state_size=state_size, action_size=action_size)
    dqn2 = DQN(state_size=state_size, action_size=action_size)

    grid = Grid(
        width,
        height,
        [
            Agent1(dqn1, buffer1),
            Agent1(dqn1, buffer1),
            Agent2(dqn2, buffer2),
            Agent2(dqn2, buffer2),
        ],
        storage,
    )
    # grid.try_load_dqn()
    grid.reset()

    grid.train(1500)
    grid.save_dqn()
    fig2, ax2 = plt.subplots()
    MLGraph(storage.ml_losses, fig2, ax2).show()

    fig1, ax1 = plt.subplots()
    vis = Visualization(fig1, ax1)
    vis.bind(storage, grid).show()
