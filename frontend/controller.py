from multiprocessing import Array
from typing import TYPE_CHECKING

from .multithread import get_process, get_test_process, get_np_from_name

if TYPE_CHECKING:
    from .model import Model


class Controller(object):
    # Iterate by number of games
    def __init__(self, model: "Model", max_itr):
        self.model = model
        self.timeout = 0.5
        self.auto_reset = True
        self.itr = 0
        self.max_itr = max_itr

        self.iterations = Array("i", range(max_itr))
        self.losses = Array("i", max_itr)
        self.epsilon = Array("f", max_itr)

        self.test_loss = Array("f", max_itr)

    def toggle_auto_reset(self):
        self.auto_reset = not self.auto_reset
        return self.auto_reset

    def next(self):
        if self.model.has_ended() and self.auto_reset:
            self.model.reset()
        self.model.step(learn=False)
        return

    def train(self, itr=1):
        self.model.reset()
        for _ in range(itr):
            (
                loss,
                reward,
                epsilon,
            ) = self.model.train_one_game()
            if self.itr >= self.max_itr:
                self.itr = 0
            self.losses[self.itr] = loss
            self.epsilon[self.itr] = epsilon
            self.itr += 1

    def test(self, itr=1):
        self.model.reset()
        for i in range(self.max_itr):
            self.test_loss[i] = 0
        for _ in range(itr):
            (
                loss,
                reward,
                epsilon,
            ) = self.model.train_one_game(learn=False)
            if self.itr >= self.max_itr:
                self.itr = 0
            self.test_loss[self.itr] = loss
            self.itr += 1

    def get_metrics(self):
        return self.iterations, self.losses, self.epsilon

    def test_in_background(self):
        gp, tp = get_test_process(self)
        gp.start()
        tp.start()
        gp.join()
        tp.join()

    def train_in_background(self):
        gp, tp, conn1 = get_process(self.model, self)
        gp.start()
        tp.start()
        gp.join()
        tp.join()

        name = conn1.recv()
        trained_Q = get_np_from_name(name)
        self.model.agent[0].Q = trained_Q
