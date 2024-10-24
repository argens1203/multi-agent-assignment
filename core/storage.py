from multiprocessing import Array
from typing import List


class Storage:
    def __init__(self, max_itr: int):
        self.itr = 0
        self.max_itr = max_itr

        # TODO: max_itr could be dynamic
        self.iterations = Array("i", range(max_itr))
        self.excess_step = Array("i", max_itr)
        self.epsilon = Array("f", max_itr)
        self.test_loss = []
        self.ml_losses = []

        self.step_count = Array("i", 51)
        self.excess_step_hist = Array("i", 51)

    def reset_counter(self):
        self.itr = 0

    def append_excess_epsilon(self, excess_step: int, epsilon: float):
        if self.itr >= self.max_itr:
            self.itr = 0
        self.excess_step[self.itr] = excess_step
        self.epsilon[self.itr] = epsilon
        self.itr += 1

    def append_step_count(self, step_count: int):
        self.step_count[step_count] += 1

    def append_excess_step_hist(self, excess_step: int):
        self.excess_step_hist[excess_step] += 1

    def append_test_loss(self, test_loss: int):
        self.test_loss.append(test_loss)

    def reset_test_loss(self):
        self.test_loss = []

    def append_ml_losses(self, ml_losses: float):
        self.ml_losses.append(ml_losses)

    def get_all(self):
        return self.iterations, self.losses, self.epsilon, self.test_loss
