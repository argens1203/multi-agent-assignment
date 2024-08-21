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
        self.grid = self.init_grid()

        self.agent = A1Agent()
        self.grid.add_agent(self.agent)

        self.costs = []

    def init_grid(self):
        grid = A1Grid(self.height, self.width)
        grid.add_goal()
        grid.add_item()
        return grid

    def get_repr(self):
        rep = self.grid.get_repr()
        print(rep)
        return rep

    # TODO: epsilon is explore ratio or exploit ratio?
    def step(self, epsilon=1):
        while not self.grid.has_ended():
            # self.agent.perceive(self.grid)
            action = self.agent.react(self.grid, epsilon)
            reward = self.grid.progress(action)
            self.agent.receive(reward)

        self.costs.append(self.grid.get_final_cost())

    def get_metrics(self):
        return self.costs

    def plot_metrics(self):
        pass
