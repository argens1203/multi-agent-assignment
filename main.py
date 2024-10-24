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

import random, os, numpy as np, torch

if __name__ == "__main__":

    storage = Storage(20000)
    possible_actions = [(0, -1), (0, 1), (-1, 0), (1, 0), (0, 0)]

    def seed_all(seed=1029):
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    seed_all(seed=1234)

    max_exp = 100
    upd_freq = 80
    eps_decay = 0.9995
    gamma = 0.9
    eps_min = 0.005
    batch_size = 32
    eps_decay_final_step = 1.99e4
    max_grad_norm = 5e3

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
    # grid.train(
    #     20000,
    #     upd_freq=upd_freq,
    #     eps_min=eps_min,
    #     eps_decay_final_step=eps_decay_final_step,
    #     max_grad_norm=max_grad_norm,
    #     dqn1=dqn1,
    #     dqn2=dqn2,
    # )

    # grid.train(20000)
    # grid.save_dqn()
    # grid.test(agents=agents)
    # grid.small_test(10000)

    fig2, ax2 = plt.subplots()
    MLGraph(storage.ml_losses, fig2, ax2).show()
    fig, axs = plt.subplots(1, 2)
    Graph(storage=storage, fig=fig, axs=(axs))

    print(storage.epsilon)

    # fig1, ax1 = plt.subplots()
    grid.reset()
    vis = Visualization(storage, grid).show()
