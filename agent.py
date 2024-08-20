from abc import ABC, abstractmethod


class Agent(ABC):
    @abstractmethod
    def perceive(self):
        pass

    @abstractmethod
    def react(self):
        pass


class GridAgent(Agent):
    def __init__(self, type, y, x):
        self.type = type
        self.y = y
        self.x = x

    def get_type(self):
        return self.type

    def get_y(self):
        return self.y

    def set_y(self, y):
        self.y = y

    def get_x(self):
        return self.x

    def set_x(self, x):
        self.x = x

    @abstractmethod
    def perceive(self, grid):
        pass

    @abstractmethod
    def react(self, grid):
        pass
