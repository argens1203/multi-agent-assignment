from abc import ABC, abstractmethod


class World(ABC):
    @abstractmethod
    def init(self):
        pass

    @abstractmethod
    def get_repr(self):
        pass

    @abstractmethod
    def get_metrics(self):
        pass

    @abstractmethod
    def step(self):
        pass

    @abstractmethod
    def plot_metrics(self):
        pass
