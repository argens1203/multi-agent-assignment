from core import (
    Agent1,
    Agent2,
    Grid,
    Storage,
    Visualization,
    ExpBuffer,
    DQN,
    Graph,
)
from constants import state_size, action_size, side

import random, os, numpy as np, torch

if __name__ == "__main__":

    storage = Storage(20000)

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
    dqn1.load("dqn1")
    dqn2.load("dqn2")
    grid.train(
        20000,
        upd_freq=upd_freq,
        eps_min=eps_min,
        eps_decay_final_step=eps_decay_final_step,
        max_grad_norm=max_grad_norm,
        dqn1=dqn1,
        dqn2=dqn2,
    )
    # dqn1.save("dqn1")
    # dqn2.save("dqn2")
    # Graph(storage=storage, keys=["ml_losses"])
    # Graph(storage=storage, keys=["excess_step", "epsilon"])

    grid.small_test(10000)
    Graph(storage=storage, keys=["excess_step_hist"])

    grid.test(1000)
    # print("---------------STEP_COUNT----------------------")
    # for i, value in enumerate(storage.step_count_hist):
    #     print(i, value)
    # print("---------------EXCESS----------------------")
    # for i, value in enumerate(storage.excess_step_hist):
    #     print(i, value)

    Graph(storage=storage, keys=["step_count_hist", "excess_step_hist"]).show()
    grid.reset()
    vis = Visualization(storage, grid).show()
