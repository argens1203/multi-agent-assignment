import random
from world import World
from .grid import A1Grid
from .agent import A1Agent


class A1World(World):
    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.agent = None
        self.grid = None
        self.init()

    def init(self):
        self.grid = A1Grid(self.height, self.width)
        self.grid.add_goal()
        self.grid.add_item()

        self.agent = A1Agent()
        self.grid.add_agent(self.agent)

    def get_repr(self):
        rep = self.grid.get_repr()
        print(rep)
        return rep

    def step(self):
        pass

    def get_metrics(self):
        pass

    def plot_metrics(self):
        pass
