from multiprocessing import Array
from typing import TYPE_CHECKING

from .multithread import get_process, get_test_process, get_np_from_name

if TYPE_CHECKING:
    from .model import Model


class Storage:
    def __init__(self, max_itr):
        self.itr = 0
        self.max_itr = max_itr

        self.iterations = Array("i", range(max_itr))
        self.losses = Array("i", max_itr)
        self.epsilon = Array("f", max_itr)
        self.test_loss = Array("f", max_itr)

    def reset_counter(self):
        self.itr = 0

    def append_loss_epsilon(self, loss, epsilon):
        if self.itr >= self.max_itr:
            self.itr = 0
        self.losses[self.itr] = loss
        self.epsilon[self.itr] = epsilon
        self.itr += 1

    def append_test_loss(self, test_loss):
        if self.itr >= self.max_itr:
            self.itr = 0
        self.test_loss[self.itr] = test_loss
        self.itr += 1

    def get_all(self):
        return self.iterations, self.losses, self.epsilon, self.test_loss


class Trainer:
    def __init__(self, model: "Model", storage: "Storage", max_itr):
        self.model = model
        self.max_itr = max_itr
        self.storage = storage

    def train(self, itr=1):
        self.model.reset()
        for _ in range(itr):
            (
                loss,
                reward,
                epsilon,
            ) = self.model.train_one_game()
            self.storage.append_loss_epsilon(loss, epsilon)

    def test(self, itr=1):
        self.model.reset()
        for i in range(self.max_itr):
            self.storage.test_loss[i] = 0
        for _ in range(itr):
            (
                loss,
                reward,
                epsilon,
            ) = self.model.train_one_game(learn=False)
            self.storage.append_test_loss(loss)

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
        trained_Q = get_np_from_name(name)
        return trained_Q


class Controller:
    # Iterate by number of games
    def __init__(self, model: "Model", max_itr):
        self.model = model
        self.auto_reset = True

        self.storage = Storage(max_itr)
        self.trainer = Trainer(self.model, self.storage, max_itr)

    def toggle_auto_reset(self):
        self.auto_reset = not self.auto_reset
        return self.auto_reset

    def next(self):
        if self.model.has_ended() and self.auto_reset:
            self.model.reset()
        self.model.step(learn=False)
        return

    def train(self, itr=1):
        self.trainer.train(itr)

    def test(self, itr=1):
        self.trainer.test(itr)

    def reset(self):
        self.model.reset()

    def get_metrics(self):
        itrs, losses, epsilons, test_losses = self.storage.get_all()
        return itrs, losses, epsilons

    def test_in_background(self, ep=1000):
        self.trainer.test_in_background(ep)

    def train_in_background(self):
        trained_Q = self.trainer.train_in_background()
        self.model.agent[0].Q = trained_Q
