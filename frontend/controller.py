from typing import TYPE_CHECKING

from .c_storage import Storage
from .c_trainer import Trainer

if TYPE_CHECKING:
    from .model import Model


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
