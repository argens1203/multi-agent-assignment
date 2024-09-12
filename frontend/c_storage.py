from multiprocessing import Array


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
