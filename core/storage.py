from multiprocessing import Array
from typing import List
import random


class Storage:
    def __init__(self, max_itr: int):
        self.itr = 0
        self.max_itr = max_itr

        # TODO: max_itr could be dynamic
        self.iterations = Array("i", range(max_itr))
        self.losses = Array("i", max_itr)
        self.epsilon = Array("f", max_itr)
        self.test_loss = []
        self.ml_losses = []
        self.step_count = []

        for i in range(max_itr):
            self.losses[i] = random.randint(0, 100)
            self.epsilon[i] = random.random()

    def reset_counter(self):
        self.itr = 0

    def append_loss_epsilon(self, loss: int, epsilon: float):
        if self.itr >= self.max_itr:
            self.itr = 0
        self.losses[self.itr] = loss
        self.epsilon[self.itr] = epsilon
        self.itr += 1

    def append_step_count(self, step_count: int):
        self.step_count.append(step_count)

    def append_test_loss(self, test_loss: int):
        self.test_loss.append(test_loss)

    def reset_test_loss(self):
        self.test_loss = []

    def append_ml_losses(self, ml_losses: List[float]):
        self.ml_losses += ml_losses

    def get_all(self):
        return self.iterations, self.losses, self.epsilon, self.test_loss
