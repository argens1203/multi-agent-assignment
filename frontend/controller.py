from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from core import Grid
    from .c_storage import Storage
    from .c_trainer import Trainer


class Controller:
    # Iterate by number of games
    def __init__(self):
        self.auto_reset = True

    def bind(self, grid: "Grid"):
        self.grid = grid
        return self

    def add_helper(self, storage: "Storage", trainer: "Trainer"):
        self.storage = storage
        self.trainer = trainer
        return self

    def toggle_auto_reset(self):
        self.auto_reset = not self.auto_reset
        return self.auto_reset

    def next(self):
        if self.grid.goal.has_reached() and self.auto_reset:
            self.grid.reset()
        self.trainer.step(learn=False)
        return

    def train(self, itr=1):
        return self.trainer.train(itr)

    def test(self, itr=1):
        return self.trainer.test(itr)

    def reset(self):
        self.grid.reset()

    def get_metrics(self):
        itrs, losses, epsilons, test_losses = self.storage.get_all()
        return itrs, losses, epsilons

    def test_in_background(self, ep=1000):
        self.trainer.test_in_background(ep)

    def train_in_background(self):
        trained_Q = self.trainer.train_in_background()
        # TODO: fix hardcode
        self.grid.agents[0].Q = trained_Q
