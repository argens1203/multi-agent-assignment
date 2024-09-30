from typing import TYPE_CHECKING, List
import datetime

from .multithread import get_process, get_test_process, get_np_from_name

if TYPE_CHECKING:
    from .c_storage import Storage
    from core import Grid, Agent


class Trainer:
    def __init__(self, max_itr):
        self.max_itr = max_itr

    def bind(self, storage: "Storage", grid: "Grid", agents: List["Agent"]):
        self.storage = storage
        self.grid = grid
        self.agents = agents

    def train(self, itr=1):
        start = datetime.datetime.now()
        print(f"Start Time: {start}")
        self.grid.reset()
        for i in range(itr):
            (loss, reward, epsilon, ml_losses) = self.train_one_game()
            # self.storage.append_loss_epsilon(loss, epsilon)

            self.storage.append_ml_losses(ml_losses)
            if (i + 1) % 100 == 0:
                print(
                    f"Epoch: {i+1}/{itr} -- Time Elapsed: {datetime.datetime.now() - start}"
                )
        return self.storage.ml_losses

    def test(self, itr=1):
        self.grid.reset()
        self.storage.reset_test_loss()
        # for i in range(self.max_itr):
        #     self.storage.test_loss[i] = 0
        for _ in range(itr):
            (loss, reward, epsilon, _) = self.train_one_game(learn=False)
            # self.storage.append_test_loss(loss)
            self.storage.append_test_loss(loss)
        return self.storage.test_loss

    def train_one_game(self, learn=True):
        self.grid.reset()
        max_reward = self.grid.get_max_reward()

        max_step_count = 50 if learn else 50
        step_count = 0
        ml_losses = []
        while not self.grid.goal.has_reached() and step_count < max_step_count:
            ml_loss = self.step(learn)
            if ml_loss is not None:
                ml_losses.append(ml_loss)
            step_count += 1

        total_reward = sum(map(lambda a: a.get_total_reward(), self.agents))
        loss = max_reward - total_reward
        return loss, total_reward, self.agents[0].epsilon, ml_losses  # TODO: 0

    def step(self, learn=True):
        return self.grid.step(learn)

    def test_in_background(self, ep=1000):
        gp, tp = get_test_process(self.storage, self, ep)
        gp.start()
        tp.start()
        gp.join()
        tp.join()

    def train_in_background(self):
        gp, tp, conn1 = get_process(self.storage, self)
        gp.start()
        tp.start()
        gp.join()
        tp.join()

        name = conn1.recv()
        # TODO: return array of trained_Q
        trained_Q = get_np_from_name(name)
        return trained_Q
