from world import World
from random import random, shuffle
from bisect import bisect

from .plot import plot_schelling
from .grid import SchellingGrid
from .agent import SchellingAgent


class SchellingWorld(World):
    def __init__(self, density, threshold, width, height):
        self.density = density
        self.height = height
        self.width = width
        self.threshold = threshold

        self.init()

    def init(self):
        self.agents = []
        self.unhappiness = []
        self.avg_similarity = []

        self.init_grid()

    def init_grid(self):
        self.grid = SchellingGrid(self.height, self.width)
        self.populate_grid(
            [self.density * 0.5, self.density, 1],
            [
                lambda y, x: SchellingAgent(1, y, x, self.threshold),
                lambda y, x: SchellingAgent(-1, y, x, self.threshold),
                lambda _, __: None,
            ],
        )
        return

    def populate_grid(self, prop_list, lambda_list):
        assert len(prop_list) == len(lambda_list)
        for x in range(self.width):
            for y in range(self.height):
                idx = bisect(prop_list, random())
                agent = lambda_list[idx](y, x)
                if agent is not None:
                    self.agents.append(agent)
                    self.grid.set_cell(agent, y, x)

    # --------- End of initialization functions --------- #

    def get_repr(self):
        rep = self.grid.get_repr()
        return rep

    def get_metrics(self):
        return self.unhappiness, self.avg_similarity

    def plot_metrics(self):
        plot_schelling(self.get_metrics())

    # --------- End of auxiliary functions --------- #

    def step(self):
        shuffle(self.agents)
        for agent in self.agents:
            agent.perceive(self.grid)
        for agent in self.agents:
            agent.react(self.grid)

        self.store_metrics()

    def store_metrics(self):
        similarities = map(lambda agent: agent.get_similarity_nearby(), self.agents)
        total_similarity = sum(similarities)
        avg_similarity = total_similarity / len(self.agents)

        unhappy_agents = filter(lambda agent: not agent.is_happy(), self.agents)
        unhappy_count = len(list(unhappy_agents))

        self.unhappiness.append(unhappy_count)
        self.avg_similarity.append(avg_similarity)
