import matplotlib.pyplot as plt

from core import (
    Agent1,
    Agent2,
    Grid,
    Storage,
    Visualization,
    MLGraph,
    ExpBuffer,
    DQN,
    Graph,
)
from constants import state_size, action_size, side

if __name__ == "__main__":

    storage = Storage(20000)
    possible_actions = [(0, -1), (0, 1), (-1, 0), (1, 0), (0, 0)]

    # min_loss = 1e9

    # for max_exp in [500]:
    #     for upd_freq in [200]:
    #         for eps_decay in [0.9995]:
    #             for gamma in [0.9]:
    #                 for batch_size in [64]:
    #                     for eps_min in [0.005]:
    #                         buffer1 = ExpBuffer(max=max_exp)
    #                         buffer2 = ExpBuffer(max=max_exp)
    #                         dqn1 = DQN(state_size=state_size, action_size=action_size)
    #                         dqn2 = DQN(state_size=state_size, action_size=action_size)
    #                         kwargs = {
    #                             "update_frequency": upd_freq,
    #                             "eps_decay": eps_decay,
    #                             "eps_min": eps_min,
    #                             "gamma": gamma,
    #                             "batch_size": batch_size,
    #                         }

    #                         grid = Grid(
    #                             side,
    #                             side,
    #                             [
    #                                 Agent1(dqn1, buffer1, **kwargs),
    #                                 Agent1(dqn1, buffer1, **kwargs),
    #                                 Agent2(dqn2, buffer2, **kwargs),
    #                                 Agent2(dqn2, buffer2, **kwargs),
    #                             ],
    #                             storage,
    #                         )
    #                         # grid.try_load_dqn()
    #                         grid.reset()
    #                         grid.train(20000)
    #                         grid.reset()
    #                         loss = grid.small_test(200)

    #                         print(
    #                             "max_exp",
    #                             max_exp,
    #                             "upd_freq",
    #                             upd_freq,
    #                             "eps_decay",
    #                             eps_decay,
    #                             "gamma",
    #                             gamma,
    #                             "batch_size",
    #                             batch_size,
    #                             "eps_min",
    #                             eps_min,
    #                         )
    #                         if loss < min_loss:
    #                             print(loss)

    max_exp = 500
    upd_freq = 200
    eps_decay = 0.9995
    gamma = 0.9
    eps_min = 0.005
    batch_size = 64

    buffer1 = ExpBuffer(max=max_exp)
    buffer2 = ExpBuffer(max=max_exp)
    dqn1 = DQN(state_size=state_size, action_size=action_size)
    dqn2 = DQN(state_size=state_size, action_size=action_size)
    kwargs = {
        "update_frequency": upd_freq,
        "eps_decay": eps_decay,
        "eps_min": eps_min,
        "gamma": gamma,
        "batch_size": batch_size,
    }

    agents = [
        Agent1(dqn1, buffer1, **kwargs),
        Agent1(dqn1, buffer1, **kwargs),
        Agent2(dqn2, buffer2, **kwargs),
        Agent2(dqn2, buffer2, **kwargs),
    ]
    grid = Grid(
        side,
        side,
        agents,
        storage,
    )
    grid.try_load_dqn()
    # grid.reset()
    # grid.train(20000)

    # grid.train(20000)
    # grid.save_dqn()
    # grid.test(agents=agents)
    grid.small_test(10000)

    fig2, ax2 = plt.subplots()
    MLGraph(storage.ml_losses, fig2, ax2).show()
    fig, axs = plt.subplots(1, 2)
    Graph(storage=storage, fig=fig, axs=(axs))

    print(storage.epsilon)

    fig1, ax1 = plt.subplots()
    grid.reset()
    vis = Visualization(fig1, ax1)
    vis.bind(storage, grid).show()
