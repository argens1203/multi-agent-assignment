from typing import TYPE_CHECKING

from .multithread import get_process, get_test_process, get_np_from_name

if TYPE_CHECKING:
    from .model import Model
    from .c_storage import Storage


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
